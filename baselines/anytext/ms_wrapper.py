'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
import torch
import random
import re
import numpy as np
import cv2
import einops
import time
from PIL import ImageFont
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from t3_dataset import draw_glyph, draw_glyph2, draw_font_hint
from cldm.recognizer import crop_image
from util import check_channels, resize_image
from safetensors import safe_open
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.models.base import TorchModel
from lora_util import get_diffusers_unet, convert_unet_state_dict_to_sd
from bert_tokenizer import BasicTokenizer
checker = BasicTokenizer()
PLACE_HOLDER = '*'
max_chars = 20


class AnyText2Model(TorchModel):

    def __init__(self, model_dir, *args, **kwargs):
        super().__init__(model_dir, *args, **kwargs)
        self.base_model_path = ''
        self.lora_paths = []
        self.lora_ratios = []
        self.use_fp16 = kwargs.get('use_fp16', True)
        self.use_translator = kwargs.get('use_translator', True)
        self.unet = get_diffusers_unet()
        self.init_model(**kwargs)

    '''
    return:
        result: list of images in numpy.ndarray format
        rst_code: 0: normal -1: error 1:warning
        str_warning: string of error or warning
        debug_info: string for debug, only valid if show_debug=True
    '''
    def forward(self, input_tensor, **forward_params):
        tic = time.time()
        str_warning = ''
        # get inputs
        seed = input_tensor.get('seed', -1)
        if seed == -1:
            seed = random.randint(0, 99999999)
        from pytorch_lightning import seed_everything
        seed_everything(seed)
        img_prompt = input_tensor.get('img_prompt')
        text_prompt = input_tensor.get('text_prompt')
        draw_pos = input_tensor.get('draw_pos')
        ori_image = input_tensor.get('ori_image')

        img_count = forward_params.get('image_count', 4)
        ddim_steps = forward_params.get('ddim_steps', 20)
        w = forward_params.get('image_width', 512)
        h = forward_params.get('image_height', 512)
        strength = forward_params.get('strength', 1.0)
        cfg_scale = forward_params.get('cfg_scale', 9.0)
        a_prompt = forward_params.get('a_prompt', 'best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
        n_prompt = forward_params.get('n_prompt', 'low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')

        img_prompt, _ = self.modify_prompt(img_prompt)
        text_prompt, texts = self.modify_prompt(text_prompt)
        if (img_prompt is None or text_prompt is None) and texts is None:
            return None, -1, "You have input Chinese prompt but the translator is not loaded!", ""
        n_lines = len(texts)

        # Simplified for inpaint ('edit') mode only
        if draw_pos is None or ori_image is None:
            return None, -1, "Reference image and position image are needed for text editing!", ""
        if isinstance(ori_image, str):
            ori_image = cv2.imread(ori_image)[..., ::-1]
            assert ori_image is not None, f"Can't read ori_image image from{ori_image}!"
        elif isinstance(ori_image, torch.Tensor):
            ori_image = ori_image.cpu().numpy()
        else:
            assert isinstance(ori_image, np.ndarray), f'Unknown format of ori_image: {type(ori_image)}'
        edit_image = ori_image.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=1024)  # make w h multiple of 64, resize if w or h > max_length
        h, w = edit_image.shape[:2]  # change h, w by input ref_img

        # preprocess pos_imgs(if numpy, make sure it's white pos in black bg)
        if isinstance(draw_pos, str):
            draw_pos = cv2.imread(draw_pos)[..., ::-1]
            assert draw_pos is not None, f"Can't read draw_pos image from{draw_pos}!"
            pos_imgs = 255-draw_pos
        elif isinstance(draw_pos, torch.Tensor):
            pos_imgs = draw_pos.cpu().numpy()
        else:
            assert isinstance(draw_pos, np.ndarray), f'Unknown format of draw_pos: {type(draw_pos)}'
            pos_imgs = draw_pos

        pos_imgs = cv2.resize(pos_imgs, (w, h))
        pos_imgs = pos_imgs[..., 0:1]
        pos_imgs = cv2.convertScaleAbs(pos_imgs)
        _, pos_imgs = cv2.threshold(pos_imgs, 254, 255, cv2.THRESH_BINARY)
        # seprate pos_imgs
        pos_imgs = self.separate_pos_imgs(pos_imgs, '↕') # Default sort_priority
        if len(pos_imgs) == 0:
            pos_imgs = [np.zeros((h, w, 1))]
        if len(pos_imgs) < n_lines:
            if n_lines == 1 and texts[0] == ' ':
                pass  # text-to-image without text
            else:
                return None, -1, f'Found {len(pos_imgs)} positions that < needed {n_lines} from prompt, check and try again!', ''
        elif len(pos_imgs) > n_lines:
            str_warning = f'Warning: found {len(pos_imgs)} positions that > needed {n_lines} from prompt.'
        # get pre_pos, poly_list, hint that needed for anytext
        pre_pos = []
        poly_list = []
        for input_pos in pos_imgs:
            if input_pos.mean() != 0:
                input_pos = input_pos[..., np.newaxis] if len(input_pos.shape) == 2 else input_pos
                poly, pos_img = self.find_polygon(input_pos)
                pre_pos += [pos_img/255.]
                poly_list += [poly]
            else:
                pre_pos += [np.zeros((h, w, 1))]
                poly_list += [None]
        np_hint = np.sum(pre_pos, axis=0).clip(0, 1)
        # prepare info dict
        info = {}
        info['glyphs'] = []
        info['gly_line'] = []
        info['positions'] = []
        info['n_lines'] = [len(texts)]*img_count
        font_hint = []
        font_paths = ['None' for i in range(len(texts))]
        info['colors'] = [np.array([500, 500, 500]) for i in range(len(texts))]

        gly_pos_imgs = []
        for i in range(len(texts)):
            text = texts[i]
            if len(text) > max_chars:
                str_warning = f'"{text}" length > max_chars: {max_chars}, will be cut off...'
                text = text[:max_chars]
            gly_scale = self.model.control_model.glyph_scale
            if pre_pos[i].mean() != 0:
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(self.font, text, poly_list[i], info['colors'][i], scale=gly_scale, width=w, height=h, add_space=True)
                
                font_hint_line = draw_glyph2(font_paths[i], text, poly_list[i], np.array([255, 255, 255]), scale=1, width=w, height=h, add_space=True)
                gly_pos_img = cv2.drawContours(glyphs*255, [poly_list[i]*gly_scale], 0, (255, 255, 255), 1)

                gly_pos_imgs += [gly_pos_img]  # for show
            else:
                glyphs = np.zeros((h*gly_scale, w*gly_scale, 3))
                gly_line = np.zeros((80, 512, 1))
                gly_pos_imgs += [np.zeros((h*gly_scale, w*gly_scale, 1))]  # for show
                font_hint_line = np.zeros((h, w, 3))
            pos = pre_pos[i][..., 0:1]
            info['glyphs'] += [self.arr2tensor(glyphs, img_count)]
            info['gly_line'] += [self.arr2tensor(gly_line, img_count)]
            info['positions'] += [self.arr2tensor(pos, img_count)]
            info['colors'][i] = self.arr2tensor(info['colors'][i], img_count)/255.
            font_hint += [font_hint_line]
        
        # get masked_x
        masked_img = ((edit_image.astype(np.float32) / 127.5) - 1.0 - np_hint*10).clip(-1, 1)
        masked_img = np.transpose(masked_img, (2, 0, 1))
        masked_img = torch.from_numpy(masked_img.copy()).float().cuda(0)
        if self.use_fp16:
            masked_img = masked_img.half()
        encoder_posterior = self.model.encode_first_stage(masked_img[None, ...])
        masked_x = self.model.get_first_stage_encoding(encoder_posterior).detach()
        if self.use_fp16:
            masked_x = masked_x.half()
        info['masked_x'] = torch.cat([masked_x for _ in range(img_count)], dim=0)

        hint = self.arr2tensor(np_hint, img_count)

        font_hint_fg = np.sum(font_hint, axis=0).clip(0, 1)[..., 0:1]*255
        font_hint_bg = font_hint_fg # Simplified font hollow logic

        info['font_hint'] = self.arr2tensor((font_hint_bg/255), img_count)
        cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[img_prompt + ', ' + a_prompt] * img_count, [text_prompt] * img_count]], text_info=info))
        un_cond = self.model.get_learned_conditioning(dict(c_concat=[hint], c_crossattn=[[[n_prompt] * img_count, [""] * img_count]], text_info=info))

        shape = (4, h // 8, w // 8)
        self.model.control_scales = ([strength] * 13)
        self.model.attnx_scale = 1.0 # Default value
        samples, intermediates = self.ddim_sampler.sample(ddim_steps, img_count,
                                                          shape, cond, verbose=False, eta=0.0,
                                                          unconditional_guidance_scale=cfg_scale,
                                                          unconditional_conditioning=un_cond)
        self.model.embedding_manager.font_hint_mimic_imgs = None  # reset mimic imgs
        if self.use_fp16:
            samples = samples.half()
        x_samples = self.model.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
        results = [x_samples[i] for i in range(img_count)]
        
        input_prompt = img_prompt + ', ' + text_prompt
        for t in texts:
            input_prompt = input_prompt.replace('*', f'"{t}"', 1)
        print(f'Prompt: {input_prompt}')
        
        # Simplified debug_info
        debug_info = ''
        rst_code = 1 if str_warning else 0
        return results, rst_code, str_warning, debug_info

    def load_weights(self):
        self.model.load_state_dict(load_state_dict(self.ckpt_path, location='cuda'), strict=False)
        print('Original weights loaded!')

    def init_model(self, **kwargs):
        if self.use_translator:
            self.trans_pipe = pipeline(task=Tasks.translation, model=os.path.join(self.model_dir, 'nlp_csanmt_translation_zh2en'))
            print(self.trans_pipe(input='初始化翻译器')['translation'])
        else:
            self.trans_pipe = None
        font_path = kwargs.get('font_path', 'font/Arial_Unicode.ttf')
        self.font = ImageFont.truetype(font_path, size=60)
        cfg_path = kwargs.get('cfg_path', 'models_yaml/anytext2_sd15.yaml')
        self.ckpt_path = kwargs.get('model_path', os.path.join(self.model_dir, 'anytext_v2.0.ckpt'))
        clip_path = os.path.join(self.model_dir, 'clip-vit-large-patch14')
        self.model = create_model(cfg_path, cond_stage_path=clip_path, use_fp16=self.use_fp16)
        if self.use_fp16:
            self.model = self.model.half()
        self.load_weights()

        self.model.eval()
        self.ddim_sampler = DDIMSampler(self.model)

    def modify_prompt(self, prompt):
        prompt = prompt.replace('“', '"')
        prompt = prompt.replace('”', '"')
        p = '"(.*?)"'
        strs = re.findall(p, prompt)
        if len(strs) == 0:
            strs = [' ']
        else:
            for s in strs:
                prompt = prompt.replace(f'"{s}"', f'{PLACE_HOLDER}', 1)
        if self.is_chinese(prompt):
            if self.trans_pipe is None:
                return None, None
            old_prompt = prompt
            prompt = self.trans_pipe(input=prompt + ' .')['translation'][:-1]
            prompt = prompt.replace(f'{PLACE_HOLDER}', f' {PLACE_HOLDER} ')
            print(f'Translate: {old_prompt} --> {prompt}')
        return prompt, strs

    def is_chinese(self, text):
        text = checker._clean_text(text)
        for char in text:
            cp = ord(char)
            if checker._is_chinese_char(cp):
                return True
        return False

    def separate_pos_imgs(self, img, sort_priority, gap=102):
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)
        components = []
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area < 20:
                continue
            component = np.zeros_like(img)
            component[labels == label] = 255
            components.append((component, centroids[label]))
        if sort_priority == '↕':
            fir, sec = 1, 0  # top-down first
        elif sort_priority == '↔':
            fir, sec = 0, 1  # left-right first
        components.sort(key=lambda c: (c[1][fir]//gap, c[1][sec]//gap))
        sorted_components = [c[0] for c in components]
        return sorted_components

    def find_polygon(self, image, min_rect=False):
        contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        max_contour = max(contours, key=cv2.contourArea)  # get contour with max area
        if min_rect:
            # get minimum enclosing rectangle
            rect = cv2.minAreaRect(max_contour)
            poly = np.int0(cv2.boxPoints(rect))
        else:
            # get approximate polygon
            epsilon = 0.01 * cv2.arcLength(max_contour, True)
            poly = cv2.approxPolyDP(max_contour, epsilon, True)
            n, _, xy = poly.shape
            poly = poly.reshape(n, xy)
        cv2.drawContours(np.ascontiguousarray(image, dtype=np.uint8), [poly], -1, 255, -1)
        return poly, image

    def arr2tensor(self, arr, bs):
        if len(arr.shape) == 3:
            arr = np.transpose(arr, (2, 0, 1))
        _arr = torch.from_numpy(arr.copy()).float().cuda(0)
        if self.use_fp16:
            _arr = _arr.half()
        _arr = torch.stack([_arr for _ in range(bs)], dim=0)
        return _arr

    # Removed load_base_model and merge_loras as they are not used in the inpainting-only flow
