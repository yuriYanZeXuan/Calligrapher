import argparse
import os
import os.path as osp
import random
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import cv2
import lightning as L
import numpy as np
from PIL import Image
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from tqdm import tqdm
import yaml

from eval.t3_dataset import T3DataSet
from src.flux.condition import Condition
from src.flux.generate_fill import generate_fill
from src.train.model import OminiModelFIll


def get_rank():
    try:
        rank = int(os.environ.get("LOCAL_RANK"))
    except:
        rank = 0
    return rank

class T3DataSetWarp(T3DataSet):
    
    to_tensor = T.ToTensor()
    def __init__(
        self,
        json_path,
        condition_size: int = 512,
        target_size: int = 512,
        glyph_scale: int = 1,
        condition_type: str = "canny",
        drop_text_prob: float = 0.1,
        drop_image_prob: float = 0.1,
        return_pil_image: bool = False,
        use_filter = False,
    ):
        self.condition_size = condition_size
        self.target_size = target_size
        self.condition_type = condition_type
        self.drop_text_prob = drop_text_prob
        self.drop_image_prob = drop_image_prob
        self.return_pil_image = return_pil_image

        self.to_tensor = T.ToTensor()

        # json_root = osp.join(osp.dirname(json_path), 'imgs')
        json_paths = [
            # [json_path, json_root]
            json_path,
            ]
        
        mask_ratio = 0  # default 0.5, ratio of mask for inpainting(text editing task), set 0 to disable
        dataset_percent = 1.0  # 1.0 use full datasets, 0.0566 use ~200k images for ablation study
        wm_thresh = 0.5  # set 0.5 to skip watermark imgs from training(ch:~25%, en:~8%, @Precision93.67%+Recall88.80%), 1.0 not skip
        super().__init__(json_paths, max_lines=8, max_chars=20, caption_pos_prob=0.0, mask_pos_prob=1.0, mask_img_prob=mask_ratio, glyph_scale=glyph_scale, percent=dataset_percent, debug=False, using_dlc=False, wm_thresh=wm_thresh, use_filter=use_filter)

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

        img_path = item_dict['img_path']
        image = item_dict['img']
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

        if image.shape[0] != self.target_size or image.shape[1] != self.target_size:
            image = cv2.resize(image, (self.target_size, self.target_size), interpolation=cv2.INTER_LINEAR)
        if hint.shape[0] != self.target_size or hint.shape[1] != self.target_size:
            hint = cv2.resize(hint, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)
        if attn_mask.shape[0] != self.target_size or attn_mask.shape[1] != self.target_size:
            attn_mask = cv2.resize(attn_mask, (self.target_size, self.target_size), interpolation=cv2.INTER_NEAREST)

        # Randomly drop text or image
        drop_text = random.random() < self.drop_text_prob
        drop_image = random.random() < self.drop_image_prob
        if drop_text:
            description = ""
        if drop_image:
            condition_img = Image.new(
                "RGB", (self.condition_size, self.condition_size), (0, 0, 0)
            )
            condition_img = np.array(condition_img)
        else:
            condition_img = glyph_img

        return {
            "img_path": img_path,
            "image": self.to_tensor(image),
            # "condition": self.to_tensor(condition_img),
            # "hint": self.to_tensor(hint),
            # "image": image,
            "condition": condition_img,
            "hint": hint,
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

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_path",
        type=str,
        default='/home/lr264907/.cache/modelscope/hub/datasets/iic/AnyWord-3M/laion/data_v1.1.json',
        help="json path for evaluation dataset",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='result/glyph',
        help="output path, clear the folder if exist",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default='models/anytext_v1.1.ckpt',
        help='path of model'
    )
    parser.add_argument(
        '--config_path',
        type=str,

    )
    parser.add_argument(
        '--target_size',
        type=int,
    )
    parser.add_argument(
        '--use_filter',
        action='store_true',
    )
    args = parser.parse_args()
    return args

def init_pipeline(args, config):
    training_config = config["train"]

    trainable_model = OminiModelFIll(
            flux_pipe_id=config["flux_path"],
            lora_config=training_config["lora_config"],
            device=f"cuda",
            dtype=getattr(torch, config["dtype"]),
            optimizer_config=training_config["optimizer"],
            model_config=config.get("model", {}),
            gradient_checkpointing=training_config.get("gradient_checkpointing", False),
            byt5_encoder_config=training_config.get("byt5_encoder", None),
        )
    
    from safetensors.torch import load_file
    state_dict = load_file(args.model_path)
    state_dict_new = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
    trainable_model.transformer.load_state_dict(state_dict_new, strict=False)

    pipe = trainable_model.flux_pipe
    
    return pipe, trainable_model

def load_dataset(args, condition_size, target_size):
    if target_size == 1024:
        glyph_scale = 2
    else:
        glyph_scale = 1
    dataset = T3DataSetWarp(
            args.json_path,
            condition_size=condition_size,
            target_size=target_size,
            glyph_scale=glyph_scale,
            condition_type='word_fill',
            drop_text_prob=0,
            drop_image_prob=0,
            use_filter=args.use_filter,
        )
    return dataset

def main():
    dist.init_process_group(backend='nccl')
    local_rank = dist.get_rank() % torch.cuda.device_count()
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    
    args = parse_args()
    if args.target_size is None:
        condition_size = 512
        target_size = 512
    elif args.target_size != 512:
        target_size = args.target_size
        condition_size = args.target_size
    else:
        condition_size = 512
        target_size = 512
    with open(args.config_path, "r") as f:
        config = yaml.safe_load(f)

    pipe, trainable_model = init_pipeline(args, config)
    dataset = load_dataset(args, condition_size, target_size)
    print("Dataset length:", len(dataset))
    train_sampler = DistributedSampler(dataset)
    train_loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=train_sampler,
    )
    generator = torch.Generator(device="cuda")

    dist.barrier()
    if local_rank == 0:
        pbar = tqdm(total=len(dataset))
    for i, data in enumerate(train_loader):
        generator.manual_seed(42)
        img = data["image"]
        condition_img = data["condition"][0].numpy()
        hint = data["hint"][0].numpy()
        condition_type = data["condition_type"][0]
        prompt = data["description"]
        img_path = data['img_path']

        condition_img = [condition_img, hint, img]
        position_delta = [0, 0]
        condition = Condition(
                        condition_type=condition_type,
                        condition=condition_img,
                        position_delta=position_delta,
                    )
        res = generate_fill(
            pipe,
            prompt=prompt,
            conditions=[condition],
            height=target_size,
            width=target_size,
            generator=generator,
            model_config=config.get("model", {}),
            default_lora=True,
        )
        img_name = img_path[0].split('.')[0]+f'_0' + '.jpg'
        img_name = osp.basename(img_name)
        os.makedirs(args.output_dir, exist_ok=True)

        res.images[0].save(
            os.path.join(args.output_dir, img_name)
        )

        if local_rank == 0:
            pbar.update(1)
    if local_rank == 0:
        pbar.close()
    dist.barrier()

if __name__ == '__main__':
    main()
