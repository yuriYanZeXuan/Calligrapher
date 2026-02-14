import math
import os
import random
import time

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from .dataset_util import load, show_bbox_on_image


phrase_list = [
    ', content and position of the texts are ',
    ', textual material depicted in the image are ',
    ', texts that says ',
    ', captions shown in the snapshot are ',
    ', with the words of ',
    ', that reads ',
    ', the written materials on the picture: ',
    ', these texts are written on it: ',
    ', captions are ',
    ', content of the text in the graphic is '
]


def insert_spaces(string, nSpace):
    if nSpace == 0:
        return string
    new_string = ""
    for char in string:
        new_string += char + " " * nSpace
    return new_string[:-nSpace]


def draw_glyph(font, text):
    g_size = 50
    W, H = (512, 80)
    new_font = font.font_variant(size=g_size)
    img = Image.new(mode='1', size=(W, H), color=0)
    draw = ImageDraw.Draw(img)
    left, top, right, bottom = new_font.getbbox(text)
    text_width = max(right-left, 5)
    text_height = max(bottom - top, 5)
    ratio = min(W*0.9/text_width, H*0.9/text_height)
    new_font = font.font_variant(size=int(g_size*ratio))

    text_width, text_height = new_font.getsize(text)
    # left, top, right, bottom = new_font.getbbox(text)
    # text_width = right - left
    # text_height = bottom - top

    offset_x, offset_y = new_font.getoffset(text)
    x = (img.width - text_width) // 2
    y = (img.height - text_height) // 2 - offset_y//2
    draw.text((x, y), text, font=new_font, fill='white')
    img = np.expand_dims(np.array(img), axis=2).astype(np.float64)
    return img


def draw_glyph2(font, text, polygon, vertAng=10, scale=1, width=512, height=512, add_space=True):
    enlarge_polygon = polygon*scale
    rect = cv2.minAreaRect(enlarge_polygon)
    box = cv2.boxPoints(rect)
    box = np.int0(box)

    w, h = rect[1]
    angle = rect[2]
    if angle < -45:
        angle += 90
    angle = -angle
    if w < h:
        angle += 90

    vert = False
    if (abs(angle) % 90 < vertAng or abs(90-abs(angle) % 90) % 90 < vertAng):
        _w = max(box[:, 0]) - min(box[:, 0])
        _h = max(box[:, 1]) - min(box[:, 1])
        if _h >= _w:
            vert = True
            angle = 0

    img = np.zeros((height*scale, width*scale, 3), np.uint8)
    img = Image.fromarray(img)

    # infer font size
    image4ratio = Image.new("RGB", img.size, "white")
    draw = ImageDraw.Draw(image4ratio)
    _, _, _tw, _th = draw.textbbox(xy=(0, 0), text=text, font=font)
    text_w = min(w, h) * (_tw / _th)

    if text_w <= max(w, h):
        # add space
        if len(text) > 1 and not vert and add_space:
            for i in range(1, 100):
                text_space = insert_spaces(text, i)
                _, _, _tw2, _th2 = draw.textbbox(xy=(0, 0), text=text_space, font=font)
                if min(w, h) * (_tw2 / _th2) > max(w, h):
                    break
            text = insert_spaces(text, i-1)
        font_size = min(w, h)*0.80
    else:
        shrink = 0.75 if vert else 0.85
        font_size = min(w, h) / (text_w/max(w, h)) * shrink
    new_font = font.font_variant(size=int(font_size))

    left, top, right, bottom = new_font.getbbox(text)
    text_width = right-left
    text_height = bottom - top

    layer = Image.new('RGBA', img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    if not vert:
        draw.text((rect[0][0]-text_width//2, rect[0][1]-text_height//2-top), text, font=new_font, fill=(255, 255, 255, 255))
    else:
        x_s = min(box[:, 0]) + _w//2 - text_height//2
        y_s = min(box[:, 1])
        for c in text:
            draw.text((x_s, y_s), c, font=new_font, fill=(255, 255, 255, 255))
            _, _t, _, _b = new_font.getbbox(c)
            y_s += _b

    rotated_layer = layer.rotate(angle, expand=1, center=(rect[0][0], rect[0][1]))

    x_offset = int((img.width - rotated_layer.width) / 2)
    y_offset = int((img.height - rotated_layer.height) / 2)
    img.paste(rotated_layer, (x_offset, y_offset), rotated_layer)
    img = np.expand_dims(np.array(img.convert('1')), axis=2).astype(np.float64)
    return img


def get_caption_pos(ori_caption, pos_idxs, prob=1.0, place_holder='*'):
    idx2pos = {
        0: " top left",
        1: " top",
        2: " top right",
        3: " left",
        4: random.choice([" middle", " center"]),
        5: " right",
        6: " bottom left",
        7: " bottom",
        8: " bottom right"
    }
    new_caption = ori_caption + random.choice(phrase_list)
    pos = ''
    for i in range(len(pos_idxs)):
        if random.random() < prob and pos_idxs[i] > 0:
            pos += place_holder + random.choice([' located', ' placed', ' positioned', '']) + random.choice([' at', ' in', ' on']) + idx2pos[pos_idxs[i]] + ', '
        else:
            pos += place_holder + ' , '
    pos = pos[:-2] + '.'
    new_caption += pos
    return new_caption


def generate_random_rectangles(w, h, box_num):
    rectangles = []
    for i in range(box_num):
        x = random.randint(0, w)
        y = random.randint(0, h)
        w = random.randint(16, 256)
        h = random.randint(16, 96)
        angle = random.randint(-45, 45)
        p1 = (x, y)
        p2 = (x + w, y)
        p3 = (x + w, y + h)
        p4 = (x, y + h)
        center = ((x + x + w) / 2, (y + y + h) / 2)
        p1 = rotate_point(p1, center, angle)
        p2 = rotate_point(p2, center, angle)
        p3 = rotate_point(p3, center, angle)
        p4 = rotate_point(p4, center, angle)
        rectangles.append((p1, p2, p3, p4))
    return rectangles


def rotate_point(point, center, angle):
    # rotation
    angle = math.radians(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    x1 = x * math.cos(angle) - y * math.sin(angle)
    y1 = x * math.sin(angle) + y * math.cos(angle)
    x1 += center[0]
    y1 += center[1]
    return int(x1), int(y1)


class T3DataSet(Dataset):
    def __init__(
            self,
            json_path,
            max_lines=5,
            max_chars=20,
            place_holder='*',
            font_path='./font/Arial_Unicode.ttf',
            caption_pos_prob=1.0,
            mask_pos_prob=1.0,
            mask_img_prob=0.5,
            for_show=False,
            using_dlc=False,
            glyph_scale=1,
            percent=1.0,
            debug=False,
            wm_thresh=1.0,
            random_select=False,
            ):
        assert isinstance(json_path, (str, list))
        if isinstance(json_path, str):
            json_path = [json_path]
        data_list = []
        self.using_dlc = using_dlc
        self.max_lines = max_lines
        self.max_chars = max_chars
        self.place_holder = place_holder
        self.font = ImageFont.truetype(font_path, size=60)
        self.caption_pos_porb = caption_pos_prob
        self.mask_pos_prob = mask_pos_prob
        self.mask_img_prob = mask_img_prob
        self.for_show = for_show
        self.glyph_scale = glyph_scale
        self.wm_thresh = wm_thresh
        for jp in json_path:
            data_list += self.load_data(jp, percent)
        self.data_list = data_list
        self.orgranize_data()
        print(f'All dataset loaded, imgs={len(self.data_list)}')
        self.debug = debug
        self.random_select = random_select
        if self.debug:
            self.tmp_items = [i for i in range(100)]

    def orgranize_data(self):
        content = {
            'path': [],
            'id': [],
            'height': [],
            'width': [],
            'fps': [],
            'num_frames': [],
        }
        for i, data in enumerate(self.data_list):
            content['path'].append(data['img_path'])
            content['id'].append(i)
            content['height'].append(data['height'])
            content['width'].append(data['width'])
            content['fps'].append(1)
            content['num_frames'].append(1)
        self.data = pd.DataFrame(content)
        self.fps_max = 1

    def is_vertical_screen(self, ploygon):
        ploygon = np.array(ploygon)
        # rect = cv2.minAreaRect(ploygon)
        # w, h = rect[1]
        x, y, w, h = cv2.boundingRect(ploygon)
        if h > 2*w:
            return True
        else:
            return False
    
    def isvalid_img(self, anns):
        valid_img = True
        for ann in anns:
            if self.is_vertical_screen(ann['polygon']):
                # 纵图 认为 valid
                continue
            # 单图维度
            if ann['valid'] and not ann['recog_valid']:
                valid_img = False
                break
        return valid_img

    def load_data(self, json_path, percent):
        tic = time.time()
        if len(json_path) == 3:
            json_path, json_root, percent = json_path
        else:
            json_path, json_root = json_path
        content = load(json_path)
        d = []
        count = 0
        wm_skip = 0
        valid_skip = 0
        max_img = len(content['data_list']) * percent
        for gt in content['data_list']:
            if len(d) > max_img:
                break
            if 'wm_score' in gt and gt['wm_score'] > self.wm_thresh:  # wm_score > thresh will be skiped as an img with watermark
                wm_skip += 1
                continue
            if 'annotations' not in gt:
                continue
            # 判断图像是否是好图, 使用参考图的不过滤
            if True:
                if not self.isvalid_img(gt['annotations']):
                    # 不是好图，过滤
                    valid_skip += 1
                    continue
            data_root = content['data_root']
            if self.using_dlc:
                data_root = data_root.replace('/data/vdb', '/mnt/data', 1)
            data_root = json_root
            if 'use_re' in gt and gt['use_re']:
                img_path = os.path.join(data_root, gt['imroot'], gt['img_name'])
                ref_path = os.path.join(data_root, gt['reroot'], gt['img_name_re'])
            else:
                img_path = os.path.join(data_root, gt['img_name'])
                ref_path = os.path.join(data_root, gt['img_name'])
            info = {}
            info['img_path'] = img_path
            info['ref_path'] = ref_path
            info['caption'] = gt['caption'] if 'caption' in gt else ''
            if self.place_holder in info['caption']:
                count += 1
                info['caption'] = info['caption'].replace(self.place_holder, " ")
            if 'annotations' in gt:
                polygons = []
                invalid_polygons = []
                texts = []
                languages = []
                pos = []
                for annotation in gt['annotations']:
                    if len(annotation['polygon']) == 0:
                        continue
                    if len(annotation['text']) == 0:
                        continue
                    if 'valid' in annotation and annotation['valid'] is False:
                        invalid_polygons.append(annotation['polygon'])
                        continue
                    polygons.append(annotation['polygon'])
                    texts.append(annotation['text'])
                    languages.append(annotation['language'])
                    if 'pos' in annotation:
                        pos.append(annotation['pos'])
                info['polygons'] = [np.array(i) for i in polygons]
                info['invalid_polygons'] = [np.array(i) for i in invalid_polygons]
                info['texts'] = texts
                info['language'] = languages
                info['pos'] = pos
            info['width'] = gt['width']
            info['height'] = gt['height']
            d.append(info)
        print(f'{json_path} loaded, imgs={len(d)}, wm_skip={wm_skip}, valid_skip={valid_skip}, time={(time.time()-tic):.2f}s')
        if count > 0:
            print(f"Found {count} image's caption contain placeholder: {self.place_holder}, change to ' '...")
        return d

    def __getitem__(self, item):
        # fullimg_size = (512, 512)  # height, width
        item, num_frames, height, width = [int(val) for val in item.split("-")]
        fullimg_size = (height, width)

        item_dict = {}
        if self.debug:  # sample fixed items
            item = self.tmp_items.pop()
            print(f'item = {item}')
        cur_item = self.data_list[item]
        # img
        target = np.array(Image.open(cur_item['img_path']).convert('RGB'))
        ref_image = np.array(Image.open(cur_item['ref_path']).convert('RGB'))
        # fullimg_size = (target.shape[0],  target.shape[1])  # for debug
        
        if target.shape[0] != fullimg_size[0] or target.shape[1] != fullimg_size[1]:
            h_ratio = target.shape[0] / fullimg_size[0]
            w_ratio = target.shape[1] / fullimg_size[1]
            
            for i in range(len(cur_item['polygons'])):
                polygon = cur_item['polygons'][i]
                _type = polygon.dtype
                # polygon = np.array(polygon)
                polygon = polygon.reshape(-1, 2)
                polygon[:, 0] = polygon[:, 0] / w_ratio
                polygon[:, 1] = polygon[:, 1] / h_ratio
                polygon = polygon.astype(_type)
                # polygon = polygon.tolist()
                cur_item['polygons'][i] = polygon
            target = cv2.resize(target, (fullimg_size[1], fullimg_size[0]))
        else:
            for i in range(len(cur_item['polygons'])):
                polygon = cur_item['polygons'][i]
                polygon = polygon.reshape(-1, 2)
                cur_item['polygons'][i] = polygon
        if ref_image.shape[0] != fullimg_size[0] or ref_image.shape[1] != fullimg_size[1]:
            ref_image = cv2.resize(ref_image, (fullimg_size[1], fullimg_size[0]))
        item_dict['img'] = target
        item_dict['refimg'] = ref_image
        # caption
        item_dict['caption'] = cur_item['caption']
        item_dict['glyphs'] = []
        item_dict['gly_line'] = []
        item_dict['positions'] = []
        item_dict['texts'] = []
        item_dict['language'] = []
        item_dict['inv_mask'] = []
        texts = cur_item.get('texts', [])
        if len(texts) > 0:
            idxs = [i for i in range(len(texts))]
            if len(texts) > self.max_lines:
                sel_idxs = random.sample(idxs, self.max_lines)
                unsel_idxs = [i for i in idxs if i not in sel_idxs]
            else:
                sel_idxs = idxs
                unsel_idxs = []
            if len(cur_item['pos']) > 0:
                pos_idxs = [cur_item['pos'][i] for i in sel_idxs]
            else:
                pos_idxs = [-1 for i in sel_idxs]
            # random select 
            if self.random_select:
                num = random.randint(1, len(sel_idxs))
                random.shuffle(sel_idxs)
                unsel_idxs.extend(sel_idxs[num:])
                sel_idxs = sel_idxs[:num]
            item_dict['caption'] = get_caption_pos(item_dict['caption'], pos_idxs, self.caption_pos_porb, self.place_holder)
            item_dict['polygons'] = [cur_item['polygons'][i] for i in sel_idxs]
            item_dict['texts'] = [cur_item['texts'][i][:self.max_chars] for i in sel_idxs]
            item_dict['language'] = [cur_item['language'][i] for i in sel_idxs]
            # glyphs
            for idx, text in enumerate(item_dict['texts']):
                gly_line = draw_glyph(self.font, text)
                glyphs = draw_glyph2(self.font, text, item_dict['polygons'][idx], scale=self.glyph_scale,  width=fullimg_size[1], height=fullimg_size[0])
                item_dict['glyphs'] += [glyphs]
                item_dict['gly_line'] += [gly_line]
            # mask_pos
            for polygon in item_dict['polygons']:
                item_dict['positions'] += [self.draw_pos(polygon, fullimg_size=fullimg_size, prob=self.mask_pos_prob)]
        # inv_mask
        invalid_polygons = cur_item['invalid_polygons'] if 'invalid_polygons' in cur_item else []
        if len(texts) > 0:
            invalid_polygons += [cur_item['polygons'][i] for i in unsel_idxs]
        item_dict['inv_mask'] = self.draw_inv_mask(invalid_polygons, fullimg_size=fullimg_size)
        item_dict['hint'] = self.get_hint(item_dict['positions'], fullimg_size=fullimg_size)
        if random.random() < self.mask_img_prob:
            # randomly generate 0~3 masks
            box_num = random.randint(0, 3)
            boxes = generate_random_rectangles(fullimg_size[1], fullimg_size[0], box_num)
            boxes = np.array(boxes)
            pos_list = item_dict['positions'].copy()
            for i in range(box_num):
                pos_list += [self.draw_pos(boxes[i], fullimg_size=fullimg_size, prob=self.mask_pos_prob)]
            mask = self.get_hint(pos_list, fullimg_size=fullimg_size)
            masked_img = target*(1-mask)
        else:
            masked_img = np.zeros_like(target)
        item_dict['masked_img'] = masked_img

        if self.for_show:
            item_dict['img_name'] = os.path.split(cur_item['img_path'])[-1]
            return item_dict
        if len(texts) > 0:
            del item_dict['polygons']
        # padding
        n_lines = min(min(len(texts), self.max_lines), len(item_dict['texts']))
        item_dict['n_lines'] = n_lines
        n_pad = self.max_lines - n_lines
        if n_pad > 0:
            item_dict['glyphs'] += [np.zeros((fullimg_size[0]*self.glyph_scale, fullimg_size[1]*self.glyph_scale, 1))] * n_pad
            item_dict['gly_line'] += [np.zeros((80, 512, 1))] * n_pad
            item_dict['positions'] += [np.zeros((fullimg_size[0], fullimg_size[1], 1))] * n_pad
            item_dict['texts'] += [' '] * n_pad
            item_dict['language'] += [' '] * n_pad

        return item_dict

    def __len__(self):
        return len(self.data_list)

    def draw_inv_mask(self, polygons, fullimg_size):
        # img = np.zeros((512, 512))
        img = np.zeros(fullimg_size)
        for p in polygons:
            pts = p.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
        img = img[..., None]
        return img/255.

    def draw_pos(self, ploygon, fullimg_size, prob=1.0):
        # img = np.zeros((512, 512))
        img = np.zeros(fullimg_size)
        rect = cv2.minAreaRect(ploygon)
        w, h = rect[1]
        small = False
        if w < 20 or h < 20:
            small = True
        if random.random() < prob:
            pts = ploygon.reshape((-1, 1, 2))
            cv2.fillPoly(img, [pts], color=255)
            # 10% dilate / 10% erode / 5% dilatex2  5% erodex2
            random_value = random.random()
            kernel = np.ones((3, 3), dtype=np.uint8)
            if random_value < 0.7:
                pass
            elif random_value < 0.8:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.9 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=1)
            elif random_value < 0.95:
                img = cv2.dilate(img.astype(np.uint8), kernel, iterations=2)
            elif random_value < 1.0 and not small:
                img = cv2.erode(img.astype(np.uint8), kernel, iterations=2)
        img = img[..., None]
        return img/255.

    def get_hint(self, positions, fullimg_size):
        if len(positions) == 0:
            return np.zeros((fullimg_size[0], fullimg_size[1], 1))
        return np.sum(positions, axis=0).clip(0, 1)

class T3DataSetWarp(T3DataSet):
    
    to_tensor = T.ToTensor()
    def __init__(
        self,
        glyph_scale: int = 1,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.0,
        return_pil_image: bool = False,
        condition_size: int = None,
        random_select: bool = False,
    ):
        self.condition_type = condition_type
        self.glyph_scale = glyph_scale
        # self.condition_type = 'subject'
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image
        self.condition_size = condition_size

        self.bucket_class = 'Bucket'
        
        self.to_tensor = T.ToTensor()

        json_paths = [
            ['dataset/Anyword/data_text_recog_glyph/Art/data-info.json', 'AnyWord-3M/ocr_data/Art/imgs/'],
            ['dataset/Anyword/data_text_recog_glyph/COCO_Text/data-info.json', 'AnyWord-3M/ocr_data/COCO_Text/imgs/'],
            ['dataset/Anyword/data_text_recog_glyph/icdar2017rctw/data-info.json', 'AnyWord-3M/ocr_data/icdar2017rctw/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/LSVT/data-info.json', 'AnyWord-3M/ocr_data/LSVT/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/mlt2019/data-info.json', 'AnyWord-3M/ocr_data/mlt2019/imgs/'],
            ['dataset/Anyword/data_text_recog_glyph/MTWI2018/data-info.json', 'AnyWord-3M/ocr_data/MTWI2018/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/ReCTS/data-info.json', 'AnyWord-3M/ocr_data/ReCTS/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/laion/data_v1.1-info.json', 'AnyWord-3M/laion/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/wukong_1of5/data_v1.1-info.json', 'AnyWord-3M/wukong_1of5/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/wukong_2of5/data_v1.1-info.json', 'AnyWord-3M/wukong_2of5/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/wukong_3of5/data_v1.1-info.json', 'AnyWord-3M/wukong_3of5/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/wukong_4of5/data_v1.1-info.json', 'AnyWord-3M/wukong_4of5/imgs'],
            ['dataset/Anyword/data_text_recog_glyph/wukong_5of5/data_v1.1-info.json', 'AnyWord-3M/wukong_5of5/imgs'],
            ]
        
        mask_ratio = 0  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
        dataset_percent = 0.1 #0.0566  # 1.0 use full datasets, 0.0566 use ~200k images for ablation study
        wm_thresh = 0.5  # set 0.5 to skip watermark imgs from training(ch:~25%, en:~8%, @Precision93.67%+Recall88.80%), 1.0 not skip
        super().__init__(json_paths, max_lines=8, max_chars=20, caption_pos_prob=0.0, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=self.glyph_scale, percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh, random_select=random_select)

    def generate_attnmask(self, n_lines, positions):
        height, width = positions[0].shape[:2]
        mask = np.zeros((height, width))
        for i in range(n_lines):
            pos = positions[i][:, :, 0] * (i+1)
            # 保证不加重复
            zero_mask = np.where(mask==0, 1, 0)
            mask = mask + zero_mask * pos
        return mask
            

    def __getitem__(self, item):
        item_dict = super().__getitem__(item)

        image = item_dict['img']
        ref_image = item_dict['refimg']
        description = item_dict['caption']
        texts = item_dict['texts']
        for text in texts:
            description = description.replace('*', text, 1)
        
        glyphs = item_dict['glyphs']
        glyph_img = np.sum(glyphs, axis=0)
        glyph_img = np.tile(glyph_img, (1,1,3))

        hint = item_dict['hint']
        hint = np.tile(hint, (1,1,3))

        position_delta = np.array([0, 0])

        # add a mask
        attn_mask = self.generate_attnmask(item_dict['n_lines'], item_dict['positions'])
        attn_mask = attn_mask.astype('uint8')

        if self.condition_size is not None:
            if glyph_img.shape[0] != self.condition_size or glyph_img.shape[1] != self.condition_size:
                glyph_img = cv2.resize(glyph_img, (self.condition_size, self.condition_size), interpolation=cv2.INTER_LINEAR)

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (glyph_img.shape[1], glyph_img.shape[0]), (0, 0, 0)
            )
            condition_img = np.array(condition_img)
        else:
            condition_img = glyph_img

        return {
            "image": self.to_tensor(image),
            "ref_image": self.to_tensor(ref_image),
            "condition": self.to_tensor(condition_img),
            "hint": self.to_tensor(hint),
            "condition_type": self.condition_type,
            "description": description,
            "position_delta": position_delta,
            "n_lines": item_dict['n_lines'],
            "gly_line": item_dict['gly_line'],
            "language": item_dict['language'],
            "positions": item_dict['positions'],
            "texts": item_dict['texts'],
            "attnmask": attn_mask,
            **({"pil_image": [image, condition_img]} if self.return_pil_image else {}),
        }


if __name__ == '__main__':
    '''
    Run this script to show details of your dataset, such as ocr annotations, glyphs, prompts, etc.
    '''
    pass


