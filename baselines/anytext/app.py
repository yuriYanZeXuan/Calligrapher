'''
AnyText2: Visual Text Generation and Editing With Customizable Attributes
Paper: https://arxiv.org/abs/2411.15245
Code: https://github.com/tyxsspa/AnyText2
Copyright (c) Alibaba, Inc. and its affiliates.
'''
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import cv2
import gradio as gr
import numpy as np
from util import check_channels, resize_image, save_images
import argparse
from PIL import ImageColor
from ms_wrapper import AnyText2Model
from modelscope import snapshot_download



img_save_folder = 'SaveImages'
load_model = True

font_path = {
    "Arial_Unicode": "font/lang_font/Arial_Unicode.ttf",
    "阿里妈妈东方大楷": "font/lang_font/阿里妈妈东方大楷.otf",
    "仿乾隆字体": "font/lang_font/仿乾隆字体.ttf",
    "钉钉进步体": "font/lang_font/钉钉进步体.ttf",
    "淘宝买菜营销体": "font/lang_font/淘宝买菜营销体.otf",
    "站酷快乐体2016修订版": "font/lang_font/站酷快乐体2016修订版.ttf",
    "站酷庆科黄油体": "font/lang_font/站酷庆科黄油体.ttf",
    "站酷小薇LOGO体": "font/lang_font/站酷小薇LOGO体.otf",
    "BadScript": "font/lang_font/BadScript-Regular.ttf",
    "BodoniModa": "font/lang_font/BodoniModa-Italic-VariableFont_opsz,wght.ttf",
    "IndieFlower": "font/lang_font/IndieFlower-Regular.ttf",
    "Jaini": "font/lang_font/Jaini-Regular.ttf",
    "LongCang": "font/lang_font/LongCang-Regular.ttf",
    "Pacifico": "font/lang_font/Pacifico-Regular.ttf",
    "PlayfairDisplay": "font/lang_font/PlayfairDisplay-VariableFont_wght.ttf",
    "SourceHanSansCN": "font/lang_font/SourceHanSansCN-Medium.otf",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--use_fp32",
        action="store_true",
        default=False,
        help="Whether or not to use fp32 during inference."
    )
    parser.add_argument(
        "--no_translator",
        action="store_true",
        default=False,
        help="Whether or not to use the CH->EN translator, which enable input Chinese prompt and cause ~4GB VRAM."
    )
    parser.add_argument(
        "--font_path",
        type=str,
        default='font/Arial_Unicode.ttf',
        help="path of a font file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='models/anytext_v2.0.ckpt',
        help="load a specified anytext checkpoint"
    )
    args = parser.parse_args()
    return args


args = parse_args()
infer_params = {
    "use_fp16": not args.use_fp32,
    "use_translator": not args.no_translator,
    "font_path": args.font_path,
}
if args.model_path:
    infer_params['model_path'] = args.model_path
if load_model:
    model_dir = snapshot_download('iic/cv_anytext2')
    os.system(f"mkdir -p models  &&  sleep 10 && cp -r {model_dir}/*  models")
    inference = AnyText2Model(model_dir='./models', **infer_params).cuda(0)


def process(mode, img_prompt, text_prompt, sort_radio, revise_pos, f1, f2, f3, f4, f5, m1, m2, m3, m4, m5, c1, c2, c3, c4, c5, show_debug, draw_img, ref_img, ori_img, img_count, ddim_steps, w, h, strength, attnx_scale, font_hollow, cfg_scale, seed, eta, a_prompt, n_prompt):
    select_font_list = [f1, f2, f3, f4, f5]
    select_color_list = [c1, c2, c3, c4, c5]
    mimic_list = [m1, m2, m3, m4, m5]

    font_hint_image = [None] * 5
    font_hint_mask = [None] * 5
    glyline_font_path = ['None'] * 5
    text_colors = ' '.join(['500,500,500']*5)
    for idx, f in enumerate(select_font_list):
        if f == 'No Font(不指定字体)':
            pass
        elif f == 'Mimic From Image(模仿图中字体)':
            img = mimic_list[idx]
            if 'layers' in img and img['layers'][0][..., 3:].mean() > 0:
                font_hint_image[idx] = img['background'][..., :3][..., ::-1]
                font_hint_mask[idx] = img['layers'][0][..., 3:]
            else:
                font_hint_image[idx] = None
                font_hint_mask[idx] = None
        else:
            glyline_font_path[idx] = font_path[f]
    for idx, c in enumerate(select_color_list):
        if c is not None:
            strs = text_colors.split()
            if isinstance(c, str) and 'rgba' in c:
                rgb = [int(float(i)) for i in c.split('(')[-1].split(')')[0].split(',')[:3]]  # for gradio 5.X
            else:
                rgb = ImageColor.getcolor(c, "RGB")
            if list(rgb) == [0, 0, 0] or rgb == [255, 255, 255]:
                rgb = (500, 500, 500)
            rgb = ','.join([str(i) for i in list(rgb)])
            strs[idx] = rgb
            text_colors = ' '.join(strs)
    # Text Generation
    if mode == 'gen':
        # create pos_imgs
        if draw_img is not None:
            pos_imgs = 255 - draw_img['background'][..., :3]
            if 'layers' in draw_img and draw_img['layers'][0][..., :3].mean() > 0:
                if draw_img['layers'][0][..., 3].mean() != 255:
                    _pos = 255 - draw_img['layers'][0][..., 3:]
                else:
                    _pos = draw_img['layers'][0][..., :3]
                    _pos[_pos < 120] = 0
                pos_imgs = pos_imgs.astype(np.float32) + (255-_pos).astype(np.float32)
                pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        else:
            pos_imgs = np.zeros((w, h, 1))

    # Text Editing
    elif mode == 'edit':
        revise_pos = False  # disable pos revise in edit mode
        if ref_img is None or ori_img is None:
            raise gr.Error('No reference image, please upload one for edit!')
        edit_image = ori_img.clip(1, 255)  # for mask reason
        edit_image = check_channels(edit_image)
        edit_image = resize_image(edit_image, max_length=1024)
        h, w = edit_image.shape[:2]
        if isinstance(ref_img, dict) and 'layers' in ref_img and ref_img['layers'][0][..., 3:].mean() > 0:
            pos_imgs = 255 - edit_image
            edit_mask = cv2.resize(ref_img['layers'][0][..., 3:], (w, h))[..., None]
            pos_imgs = pos_imgs.astype(np.float32) + edit_mask.astype(np.float32)
            pos_imgs = pos_imgs.clip(0, 255).astype(np.uint8)
        else:
            if isinstance(ref_img, dict) and 'background' in ref_img:
                ref_img = ref_img['background'][..., :3]
            pos_imgs = 255 - ref_img  # example input ref_img is used as pos
    cv2.imwrite('pos_imgs.png', 255-pos_imgs[..., ::-1])
    params = {
        "mode": mode,
        "sort_priority": sort_radio,
        "show_debug": show_debug,
        "revise_pos": revise_pos,
        "image_count": img_count,
        "ddim_steps": ddim_steps,
        "image_width": w,
        "image_height": h,
        "strength": strength,
        "attnx_scale": attnx_scale,
        "font_hollow": font_hollow,
        "cfg_scale": cfg_scale,
        "eta": eta,
        "a_prompt": a_prompt,
        "n_prompt": n_prompt,
        "base_model_path": '',
        "lora_path_ratio": '',
        "glyline_font_path": glyline_font_path,
        "font_hint_image": font_hint_image,
        "font_hint_mask": font_hint_mask,
        "text_colors": text_colors
    }
    input_data = {
        "img_prompt": img_prompt,
        "text_prompt": text_prompt,
        "seed": seed,
        "draw_pos": pos_imgs,
        "ori_image": ori_img,
    }

    results, rtn_code, rtn_warning, debug_info = inference(input_data, **params)
    if rtn_code >= 0:
        save_images(results, img_save_folder)
        print(f'Done, result images are saved in: {img_save_folder}')
        if rtn_warning:
            gr.Warning(rtn_warning)
    else:
        raise gr.Error(rtn_warning)
    return results, gr.Markdown(debug_info, visible=show_debug)


def create_canvas(w=512, h=512, c=3, line=5):
    image = np.full((h, w, c), 200, dtype=np.uint8)
    for i in range(h):
        if i % (w//line) == 0:
            image[i, :, :] = 150
    for j in range(w):
        if j % (w//line) == 0:
            image[:, j, :] = 150
    image[h//2-8:h//2+8, w//2-8:w//2+8, :] = [200, 0, 0]
    # return image
    return {
        'background': image,
        "layers": [image],
        "composite": None
    }


def resize_w(w, img):
    if isinstance(img, dict):
        img = img['background']
    _img = cv2.resize(img, (w, img.shape[0]))
    return {
        'background': _img,
        "layers": [_img],
        "composite": None
    }


def resize_h(h, img):
    if isinstance(img, dict):
        img = img['background']
    _img = cv2.resize(img, (img.shape[1], h))
    return {
        'background': _img,
        "layers": [_img],
        "composite": None
    }


click_edit_exp = False
block = gr.Blocks(css='style.css', theme=gr.themes.Soft()).queue()


with block as demo:
    gr.HTML('<div style="text-align: center; margin: 20px auto;"> \
            <img id="banner" src="https://modelscope.cn/api/v1/studio/iic/studio_anytext2/repo?Revision=master&FilePath=example_images%2Fbanner2.jpg&View=true" style="max-width:400px; width:100%; margin:auto; display:block;" alt="anytext2"> <br>  \
            [<a href="https://arxiv.org/abs/2411.15245" style="color:blue; font-size:18px;">arXiv</a>] \
            [<a href="https://github.com/tyxsspa/AnyText2" style="color:blue; font-size:18px;">Code</a>] \
            [<a href="https://modelscope.cn/studios/iic/studio_anytext2" style="color:blue; font-size:18px;">ModelScope</a>]\
            version: 1.0.0 </div>')
    with gr.Row(variant='compact'):
        with gr.Column(scale=3) as left_part:
            pass
        with gr.Column(scale=3):
            result_gallery = gr.Gallery(label='Result(结果)', show_label=True, preview=True, columns=2, allow_preview=True, height=600)
            result_info = gr.Markdown('', visible=False)
        with left_part:
            with gr.Accordion('🛠Parameters(参数)', open=False):
                with gr.Row(variant='compact'):
                    img_count = gr.Slider(label="Image Count(图片数)", minimum=1, maximum=12, value=4, step=1)
                    ddim_steps = gr.Slider(label="Steps(步数)", minimum=1, maximum=100, value=20, step=1)
                with gr.Row(variant='compact'):
                    image_width = gr.Slider(label="Image Width(宽度)", minimum=256, maximum=1024, value=512, step=64)
                    image_height = gr.Slider(label="Image Height(高度)", minimum=256, maximum=1024, value=512, step=64)
                with gr.Row(variant='compact'):
                    strength = gr.Slider(label="Strength(控制力度)", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                    cfg_scale = gr.Slider(label="CFG-Scale(CFG强度)", minimum=0.1, maximum=30.0, value=7.5, step=0.1)
                    attnx_scale = gr.Slider(label="Attnx_Scale(控制力度)", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                with gr.Row(variant='compact'):
                    seed = gr.Slider(label="Seed(种子数)", minimum=-1, maximum=99999999, step=1, randomize=False, value=-1)
                    eta = gr.Number(label="eta (DDIM)", value=0.0)
                with gr.Row(variant='compact'):
                    show_debug = gr.Checkbox(label='Show Debug(调试信息)', value=True)
                    gr.Markdown('<span style="color:silver;font-size:12px">whether show glyph image and debug information in the result(是否在结果中显示glyph图以及调试信息)</span>')
                with gr.Row():
                    sort_radio = gr.Radio(["↕", "↔"], value='↕', label="Sort Position(位置排序)", info="position sorting priority(位置排序时的优先级)")
                    with gr.Row():
                        revise_pos = gr.Checkbox(label='Revise Position(修正位置)', value=False)
                        font_hollow = gr.Checkbox(label='Use hollow font(使用空心字体)', value=True)
                # base_model_path = gr.Textbox(label='Base Model Path(基模地址)', placeholder='/path/of/base/model')
                # lora_path_ratio = gr.Textbox(label='LoRA Path and Ratio(lora地址和比例)', placeholder='/path/of/lora1.pth ratio1 /path/of/lora2.pth ratio2 ...')
                a_prompt = gr.Textbox(label="Added Prompt(附加提示词)", value='best quality, extremely detailed,4k, HD, supper legible text,  clear text edges,  clear strokes, neat writing, no watermarks')
                n_prompt = gr.Textbox(label="Negative Prompt(负向提示词)", value='low-res, bad anatomy, extra digit, fewer digits, cropped, worst quality, low quality, watermark, unreadable text, messy words, distorted text, disorganized writing, advertising picture')
            img_prompt = gr.Textbox(label="Image Prompt(图像提示词)", placeholder="Describe details for the image, e.g.: A cartoon cat holding a sign with words on it(详细描述你要生成的图片，如：一只卡通风格的小猫举着牌子，上面写着文字)")
            text_prompt = gr.Textbox(label="Text Prompt(文字提示词)", placeholder='Write down the text, wrapping each line in double quotation marks, e.g.: that reads "Hello", "world!"(写下你要生成的文字，每行用双引号包裹，如：上面写着"你好", "世界")')

            select_font_list = []
            mimic_list = []
            select_color_list = []

            font_values = ['No Font(不指定字体)', 'Mimic From Image(模仿图中字体)'] + list(font_path.keys())
            gr.Markdown('<span style="color:silver;font-size:15px">Specify font and color of each line, random attributes will be applied if select No Font or pure black white colors.(指定每行文字的字体和颜色, 不指定字体或使用纯黑白颜色,则会使用随机属性)</span>')
            placeholder_mimic = "Upload an image and use the brush tool below to select the text area you want to mimic the font style from. It's best for that area to have the same number of characters.(上传一张图片，用下方的笔刷工具在图中选择你要模仿字体风格的文字区域, 该区域最好具有相同数量的字符)"
            with gr.Column():
                with gr.Row():
                    gr.Markdown('### 1')
                    font1 = gr.Dropdown(font_values, label="Font(字体)", interactive=True, scale=18, container=False)
                    color1 = gr.ColorPicker(label="Color(颜色)", scale=16, container=False)
                mimic_font_img1 = gr.ImageMask(sources=['upload', 'clipboard'], placeholder=placeholder_mimic, transforms=(), layers=False, brush=gr.Brush(default_size=40), visible=False)
                with gr.Row():
                    gr.Markdown('### 2')
                    font2 = gr.Dropdown(font_values, label="Font(字体)", interactive=True, scale=18, container=False)
                    color2 = gr.ColorPicker(label="Color(颜色)", scale=16, container=False)
                mimic_font_img2 = gr.ImageMask(sources=['upload', 'clipboard'], placeholder=placeholder_mimic, transforms=(), layers=False, brush=gr.Brush(default_size=40), visible=False)
                with gr.Row():
                    gr.Markdown('### 3')
                    font3 = gr.Dropdown(font_values, label="Font(字体)", interactive=True, scale=18, container=False)
                    color3 = gr.ColorPicker(label="Color(颜色)", scale=16, container=False)
                mimic_font_img3 = gr.ImageMask(sources=['upload', 'clipboard'], placeholder=placeholder_mimic, transforms=(), layers=False, brush=gr.Brush(default_size=40), visible=False)
                with gr.Row():
                    gr.Markdown('### 4')
                    font4 = gr.Dropdown(font_values, label="Font(字体)", interactive=True, scale=18, container=False)
                    color4 = gr.ColorPicker(label="Color(颜色)", scale=16, container=False)
                mimic_font_img4 = gr.ImageMask(sources=['upload', 'clipboard'], placeholder=placeholder_mimic, transforms=(), layers=False, brush=gr.Brush(default_size=40), visible=False)
                with gr.Row():
                    gr.Markdown('### 5')
                    font5 = gr.Dropdown(font_values, label="Font(字体)", interactive=True, scale=18, container=False)
                    color5 = gr.ColorPicker(label="Color(颜色)", scale=16, container=False)
                mimic_font_img5 = gr.ImageMask(sources=['upload', 'clipboard'], placeholder=placeholder_mimic, transforms=(), layers=False, brush=gr.Brush(default_size=40), visible=False)

            def sel_font(font):
                vis = font == 'Mimic From Image(模仿图中字体)'
                return gr.ImageMask(visible=vis, interactive=True)

            font1.change(fn=sel_font, inputs=[font1], outputs=[mimic_font_img1])
            font2.change(fn=sel_font, inputs=[font2], outputs=[mimic_font_img2])
            font3.change(fn=sel_font, inputs=[font3], outputs=[mimic_font_img3])
            font4.change(fn=sel_font, inputs=[font4], outputs=[mimic_font_img4])
            font5.change(fn=sel_font, inputs=[font5], outputs=[mimic_font_img5])

            select_font_list = [font1, font2, font3, font4, font5]
            select_color_list = [color1, color2, color3, color4, color5]
            mimic_list = [mimic_font_img1, mimic_font_img2, mimic_font_img3, mimic_font_img4, mimic_font_img5]

            with gr.Tabs() as tab_modes:
                with gr.Tab("🖼Text Generation(文字生成)", elem_id='MD-tab-t2i') as mode_gen:
                    gr.Markdown('<span style="color:silver;font-size:15px">Use a brush to specify the position(s) of the text, the length should be resonable(用笔刷指定每行文字的位置, 长度要与文字个数保持合理)</span>')
                    with gr.Row():
                        gr.Markdown("")
                        draw_img = gr.Sketchpad(value=None, label="Draw Position(绘制位置)", scale=3, visible=True, eraser=False, container=False, transforms=(), show_label=False, layers=False, brush=gr.Brush(default_size=40, default_color="rgb(66, 66, 66)"), )
                        gr.Markdown("")

                    def re_draw():
                        return [gr.Sketchpad(value=create_canvas(), container=False, layers=False, scale=3, eraser=False, show_label=False, brush=gr.Brush(default_size=40, default_color="rgb(66, 66, 66)")), gr.Slider(value=512), gr.Slider(value=512)]
                    draw_img.clear(re_draw, None, [draw_img, image_width, image_height])
                    image_width.release(resize_w, [image_width, draw_img], [draw_img])
                    image_height.release(resize_h, [image_height, draw_img], [draw_img])

                    with gr.Row():
                        gr.Markdown("")
                        run_gen = gr.Button(value="Run(运行)!", scale=3, elem_classes='run')
                        gr.Markdown("")

                    def exp_gen_click():
                        return [gr.Slider(value=512), gr.Slider(value=512)]  # all examples are 512x512, refresh draw_img
                    gr.Markdown('<span style="color:silver;font-size:15px">👇🏻👇🏻👇🏻Click an example and run! (点击任一示例, 运行!)</span>')
                    with gr.Tab("English Examples"):
                        exp_gen_en = gr.Examples(
                            [
                                ['photo of caramel macchiato coffee on the table, top-down perspective, with words written on it using cream', '"Any" "Text" "2"', "example_images/gen9.png", "↔", False, 4, "IndieFlower", "IndieFlower", "阿里妈妈东方大楷", 'rgba(152, 58, 16, 1)', 'rgba(155, 61, 16, 1)', 'rgba(65, 18, 6,1)', 66273235],
                                ['A raccoon stands in front of the blackboard with words written on it', 'Texts are "Deep Learning"', "example_images/gen17.png", "↕", False, 4, "阿里妈妈东方大楷", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(215, 225, 224, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 7251085],
                                ['A crayon drawing by child,  a snowman with a Santa hat, pine trees, outdoors in heavy snowfall', 'titled "Snowman"', "example_images/gen18.png", "↕", False, 4, "BadScript", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 35621187],
                                ['A fancy square birthday cake on the table, texts written in cream, close-up view, top-down perspective', 'Texts are “Generated" "by" "AnyText2"', "example_images/cake.png", "↕", False, 4, "Arial_Unicode", "IndieFlower", "BodoniModa", 'rgba(21, 254, 230, 1)', 'rgba(181, 185, 58, 1)', 'rgba(249, 100, 100,1)', 41799568],
                                ['A meticulously designed logo, a minimalist brain, stick drawing style, simplistic style,  refined with minimal strokes, black and white color, white background,  futuristic sense, exceptional design', 'logo name is "NextAI"', "example_images/gen19.png", "↕", False, 4, "IndieFlower", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 23692115],
                                ['A fine sweater with knitted text', '"Have" "A" "Good Day"', "example_images/gen20.png", "↕", False, 4, "SourceHanSansCN", "Jaini", "SourceHanSansCN", 'rgba(18, 49, 91, 1)', 'rgba(57, 177, 48, 1)', 'rgba(25, 54, 88,1)', 18752346],
                                ['Sign on the clean building with text on it', 'that reads "科学" and "과학"  and "サイエンス" and "SCIENCE"', "example_images/gen6.png", "↕", False, 4, "Arial_Unicode", "Arial_Unicode", "Arial_Unicode", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(255, 255, 255,1)', 4498087],
                                ['A nice drawing in pencil of Michael Jackson,  with words written on it', '"Micheal" and "Jackson"', "example_images/gen7.png", "↕", False, 4, "IndieFlower", "IndieFlower", "No Font(不指定字体)", 'rgba(39, 41, 33, 1)', 'rgba(41, 43, 35, 1)', 'rgba(0, 0, 0, 1)', 83866922],
                                ['a well crafted ice sculpture that made with text. Dslr photo, perfect illumination', '"Happy" and "Holidays"', "example_images/gen11.png", "↕", False, 4, "Pacifico", "Pacifico", "No Font(不指定字体)", 'rgba(211, 220, 228, 1)', 'rgba(233, 237, 241, 1)', 'rgba(249, 100, 100,1)', 91944158],
                            ],
                            [img_prompt, text_prompt, draw_img, sort_radio, revise_pos, img_count, font1, font2, font3, color1, color2, color3, seed],
                            examples_per_page=5,
                            label=''
                        )
                        exp_gen_en.dataset.click(exp_gen_click, None, [image_width, image_height])
                    with gr.Tab("中文示例"):
                        exp_gen_ch = gr.Examples(
                            [
                                ['一个精致的中国传统月饼，放在白色盘子里，月饼上面有雕刻的文字和花朵', '"中秋" "团圆"', "example_images/yuebing.png", "↕", False, 4, "Arial_Unicode", "仿乾隆字体", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 40091080],
                                ['木桌上放着一块绣着字的布和一只可爱的小老虎。布旁边有一支点燃的蜡烛。', '"晚安" "Goodnight"', "example_images/tiger.png", "↕", False, 4, "Arial_Unicode", "站酷快乐体2016修订版", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(222, 175, 0, 1)', 'rgba(0, 188, 175, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 86670233],
                                ['一只浣熊站在黑板前', '上面写着"深度学习"', "example_images/gen1.png", "↕", False, 4, "LongCang", "No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(224, 215, 215, 1)', 'rgba(0, 0, 0,1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 81808278],
                                ['一个精美的棒球帽放在木桌上，上面有针织的文字', ' "生成式模型"', "example_images/bangqiumao.png", "↕", False, 4, "阿里妈妈东方大楷", "No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 96425704],
                                ['一个儿童蜡笔画，森林里有一个可爱的蘑菇形状的房子', '标题是"森林小屋"', "example_images/gen16.png", "↕", False, 4, "SourceHanSansCN", "No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 40173333],
                                ['一个精美设计的logo，画的是一个黑白风格的厨师，带着厨师帽', 'logo下方写着“深夜食堂”', "example_images/gen14.png", "↕", False, 4, "站酷快乐体2016修订版", "No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 37000560],
                                ['一个精致的马克杯，上面雕刻着一首中国古诗', '内容是 "花落知多少" "夜来风雨声" "处处闻啼鸟" "春眠不觉晓"', "example_images/gen3.png", "↔", False, 4, "阿里妈妈东方大楷", "阿里妈妈东方大楷", "阿里妈妈东方大楷", "阿里妈妈东方大楷", 'rgba(131, 56, 31, 1)', 'rgba(132, 57, 34, 1)', 'rgba(134, 59, 37, 1)', 'rgba(136, 61, 39, 1)', 94328817],
                                ['一件精美的毛衣，上面有针织的文字', '文字内容是: "图文融合"', "example_images/gen4.png", "↕", False, 4, "Arial_Unicode", "No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(67, 225, 186, 1)', 'rgba(0, 0, 0,1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 48769450],
                                ['一个双肩包放在桌子上，近景拍摄，上面有针织的文字', '”为了无法“ ”计算的价值“', "example_images/gen12.png", "↕", False, 4, "淘宝买菜营销体", "仿乾隆字体", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(255, 201, 26, 1)', 'rgba(0, 177, 255, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)', 49171567],
                                ['一个漂亮的蜡笔画，有行星，宇航员，还有宇宙飞船', '上面写的是"去火星旅行", "王小明", "11月1日"', "example_images/gen5.png", "↕", False, 4, "仿乾隆字体", "站酷庆科黄油体", "站酷庆科黄油体", "No Font(不指定字体)", 'rgba(117, 117, 117, 1)', 'rgba(123, 152, 169, 1)', 'rgba(123, 152, 169, 1)', 'rgba(0, 0, 0,1)', 32608039],
                                ['一个装饰华丽的蛋糕，上面用奶油写着文字', '“阿里云”和"APSARA"', "example_images/gen13.png", "↕", False, 4, "阿里妈妈东方大楷", "BodoniModa", "No Font(不指定字体)", "No Font(不指定字体)", 'rgba(0, 255, 147, 1)', 'rgba(246, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 98381182],
                                ['一枚中国古代铜钱', '上面的文字是 "图"  "文" "融" "合"', "example_images/gen2.png", "↕", False, 4, "站酷庆科黄油体", "站酷庆科黄油体", "站酷庆科黄油体", "站酷庆科黄油体", 'rgba(208, 196, 152, 1)', 'rgba(221, 212, 165,1)', 'rgba(220, 213, 166, 1)', 'rgba(220, 214, 167,1)', 20842124],
                            ],
                            [img_prompt, text_prompt, draw_img, sort_radio, revise_pos, img_count, font1, font2, font3, font4, color1, color2, color3, color4, seed],
                            examples_per_page=5,
                            label=''
                        )
                        exp_gen_ch.dataset.click(exp_gen_click, None, [image_width, image_height])

                with gr.Tab("🎨Text Editing(文字编辑)") as mode_edit:
                    gr.Markdown('<span style="color:silver;font-size:15px">Tips: Mimic specific font from original image may provide better results.(提示: 从原始图像中的特定区域模仿字体风格, 可提供更好的编辑效果)</span>')
                    with gr.Row(variant='compact'):
                        ref_img = gr.ImageMask(label='Ref(参考图)', sources=['upload', 'clipboard'], scale=6, transforms=(), layers=False, brush=gr.Brush(default_size=40, default_color="rgb(66, 66, 66)"),
                                               placeholder='Upload an image and specify the area(s) you want to edit with a brush(上传一张图片，用笔刷涂抹要编辑的区域)')
                        ori_img = gr.Image(label='Ori(原图)', scale=4, container=False, interactive=False)

                    def upload_ref(ref, ori):
                        global click_edit_exp
                        if click_edit_exp:
                            original_image = gr.Image()
                            click_edit_exp = False
                        else:
                            original_image = gr.Image(value=ref['background'][..., :3])

                        return [gr.ImageMask(type="numpy", brush=gr.Brush(default_size=40, default_color="rgb(66, 66, 66)")),
                                original_image]

                    def clear_ref(ref, ori):
                        return [gr.ImageMask(), gr.Image(value=None)]
                    ref_img.upload(upload_ref, [ref_img, ori_img], [ref_img, ori_img])
                    ref_img.clear(clear_ref, [ref_img, ori_img], [ref_img, ori_img])
                    with gr.Row():
                        gr.Markdown("")
                        run_edit = gr.Button(value="Run(运行)!", scale=3, elem_classes='run')
                        gr.Markdown("")

                    def click_exp(ori_img):
                        global click_edit_exp
                        if ori_img is None:
                            click_edit_exp = True
                    gr.Markdown('<span style="color:silver;font-size:15px">Click an example to automatically fill in the parameters.(点击以下示例，自动填充参数.)</span>')
                    with gr.Tab("English Examples"):
                        defult_font_color = ["No Font(不指定字体)", "No Font(不指定字体)", "No Font(不指定字体)","No Font(不指定字体)", 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0, 1)', 'rgba(0, 0, 0,1)']
                        en_exp = gr.Examples(
                            [
                                ['A pile of fruit with words written in the middle', '"UIT"', "example_images/ref13.jpg", "example_images/edit13.png", 4] + defult_font_color +[91498555],
                                ['Characters written in chalk on the blackboard', '"DADDY"', "example_images/ref8.jpg", "example_images/edit8.png", 4] + defult_font_color +[50165756],
                                ['The blackboard with words', '"Here"', "example_images/ref11.jpg", "example_images/edit11.png", 2] + defult_font_color +[15353513],
                                ['A letter picture', '"THER"', "example_images/ref6.jpg", "example_images/edit6.png", 4] + defult_font_color +[38483041],
                                ['A cake with colorful characters', '"EVERYDAY"', "example_images/ref7.jpg", "example_images/edit7.png", 4] + defult_font_color +[8943410],
                                ['photo of clean sandy beach', '" " " "', "example_images/ref16.jpeg", "example_images/edit16.png", 4] + defult_font_color +[85664100],
                            ],
                            [img_prompt, text_prompt, ori_img, ref_img, img_count, font1, font2, font3, font4, color1, color2, color3, color4, seed],
                            examples_per_page=5,
                            label=''
                        )
                        en_exp.dataset.click(click_exp, ori_img)
                    with gr.Tab("中文示例"):
                        cn_exp = gr.Examples(
                            [
                                ['一个小猪的表情包', '"下班"', "example_images/ref2.jpg", "example_images/edit2.png", 2] + defult_font_color +[43304008],
                                ['一个中国古代铜钱', '上面写着"乾" "隆"', "example_images/ref12.png", "example_images/edit12.png", 4] + defult_font_color +[89159482],
                                ['一个漫画', '" "', "example_images/ref14.png", "example_images/edit14.png", 4] + defult_font_color +[68511317],
                                ['一个黄色标志牌', '"不要" 和 "大意"', "example_images/ref3.jpg", "example_images/edit3.png", 2] + defult_font_color +[68988613],
                                ['一个青铜鼎', '"  ", "  "', "example_images/ref4.jpg", "example_images/edit4.png", 4] + defult_font_color +[71139289],
                                ['一个建筑物前面的字母标牌', '" "', "example_images/ref5.jpg", "example_images/edit5.png", 4] + defult_font_color +[50416289],
                            ],
                            [img_prompt, text_prompt, ori_img, ref_img, img_count, font1, font2, font3, font4, color1, color2, color3, color4, seed],
                            examples_per_page=5,
                            label=''
                        )
                        cn_exp.dataset.click(click_exp, ori_img)
    ips = [img_prompt, text_prompt, sort_radio, revise_pos, *select_font_list, *mimic_list, *select_color_list, show_debug, draw_img, ref_img, ori_img, img_count, ddim_steps, image_width, image_height, strength, attnx_scale, font_hollow, cfg_scale, seed, eta, a_prompt, n_prompt]
    run_gen.click(fn=process, inputs=[gr.State('gen')] + ips, outputs=[result_gallery, result_info])
    run_edit.click(fn=process, inputs=[gr.State('edit')] + ips, outputs=[result_gallery, result_info])
    demo.load(
        fn=lambda: [gr.Sketchpad(value=create_canvas(), container=False, layers=False, scale=3,
                                 visible=True, eraser=False, show_label=False,
                                 brush=gr.Brush(default_size=40, default_color="rgb(66, 66, 66)")),
                    gr.Slider(value=512),
                    gr.Slider(value=512)],
        inputs=None,
        outputs=[draw_img, image_width, image_height])

block.launch(server_name='0.0.0.0' if os.getenv('GRADIO_LISTEN', '') != '' else "127.0.0.1", share=False)
