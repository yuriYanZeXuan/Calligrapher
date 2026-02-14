import os

import cv2
import lightning as L
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
from transformers import pipeline
import torch
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as T

try:
    import wandb
except ImportError:
    wandb = None

from ..flux.condition import Condition
from ..flux.generate_fill import generate_fill


class TrainingCallback(L.Callback):
    def __init__(self, run_name, training_config: dict = {}):
        self.run_name, self.training_config = run_name, training_config

        self.print_every_n_steps = training_config.get("print_every_n_steps", 10)
        self.save_interval = training_config.get("save_interval", 1000)
        self.sample_interval = training_config.get("sample_interval", 1000)
        self.save_path = training_config.get("save_path", "./output")

        self.wandb_config = training_config.get("wandb", None)
        self.use_wandb = (
            wandb is not None and os.environ.get("WANDB_API_KEY") is not None
        )
        if not self.use_wandb:
            self.writer = SummaryWriter(log_dir=f"{self.save_path}/{self.run_name}/logs")
        else:
            self.writer = None
        self.to_tensor = T.ToTensor()

        self.total_steps = 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        gradient_size = 0
        max_gradient_size = 0
        count = 0
        
        for _, param in pl_module.named_parameters():
            if param.grad is not None:
                gradient_size += param.grad.norm(2).item()
                max_gradient_size = max(max_gradient_size, param.grad.norm(2).item())
                count += 1
        if count > 0:
            gradient_size /= count

        self.total_steps += 1

        # Print training progress every n steps
        if self.use_wandb:
            report_dict = {
                "steps": batch_idx,
                "steps": self.total_steps,
                "epoch": trainer.current_epoch,
                "gradient_size": gradient_size,
            }
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            report_dict["loss"] = loss_value
            report_dict["t"] = pl_module.last_t
            wandb.log(report_dict)
        else:
            loss_value = outputs["loss"].item() * trainer.accumulate_grad_batches
            self.writer.add_scalar('loss/train', loss_value, self.total_steps)
            self.writer.add_scalar('t', pl_module.last_t, self.total_steps)
            self.writer.add_scalar('gradient_size', gradient_size, self.total_steps)
            self.writer.add_scalar('epoch', trainer.current_epoch, self.total_steps)
            self.writer.add_scalar('loss_sd', pl_module.res['loss_sd'].item(), self.total_steps)
            self.writer.add_scalar('loss_mask', pl_module.res['loss_mask'].item(), self.total_steps)
            if 'loss_odm' in pl_module.res:
                self.writer.add_scalar('loss_odm', pl_module.res['loss_odm'].item(), self.total_steps)
            if 'loss_ocr' in pl_module.res:
                self.writer.add_scalar('loss_ocr', pl_module.res['loss_ocr'].item(), self.total_steps)
                self.writer.add_scalar('loss_ctc', pl_module.res['loss_ctc'].item(), self.total_steps)


        if self.total_steps % self.print_every_n_steps == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps}, Batch: {batch_idx}, Loss: {pl_module.log_loss:.4f}, Gradient size: {gradient_size:.4f}, Max gradient size: {max_gradient_size:.4f}"
            )

        # Save LoRA weights at specified intervals
        if self.total_steps % self.save_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Saving LoRA weights"
            )
            pl_module.save_lora(
                f"{self.save_path}/{self.run_name}/ckpt/{self.total_steps}"
            )

        # Generate and save a sample image at specified intervals
        if self.total_steps % self.sample_interval == 0:
            print(
                f"Epoch: {trainer.current_epoch}, Steps: {self.total_steps} - Generating a sample"
            )
            self.generate_a_sample(
                trainer,
                pl_module,
                f"{self.save_path}/{self.run_name}/output",
                f"lora_{self.total_steps}",
                batch["condition_type"][
                    0
                ],  # Use the condition type from the current batch
            )

    @torch.no_grad()
    def generate_a_sample(
        self,
        trainer,
        pl_module,
        save_path,
        file_name,
        condition_type="super_resolution",
    ):
        # TODO: change this two variables to parameters
        condition_size = trainer.training_config["dataset"].get("condition_size", 512)
        target_size = trainer.training_config["dataset"].get("target_size", 512)

        generator = torch.Generator(device=pl_module.device)
        generator.manual_seed(42)

        test_list = []

        if condition_type == 'word_fill':
            hint = Image.open("assets/hint2.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            img = Image.open("assets/hint_imgs2.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = Image.open("assets/hint_imgs_word2.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            attnmask = None
            
            hint = np.array(hint) / 255
            condition_img = np.array(condition_img)
            condition_img = (255 - condition_img) / 255
            word_prompts = ["精神食粮", " ", " ", " ", " ", " ", " ", " "]
            test_list.append(([condition_img, hint, img], [0, 0], "chinese calligraphy font with the word 'love' written in it, that reads 精神食粮 ."))
            #
            hint = Image.open("assets/hint1.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            img = Image.open("assets/hint_imgs1.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = Image.open("assets/hint_imgs_word1.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            attnmask = None
            
            hint = np.array(hint) / 255
            # img = np.array(img)
            condition_img = np.array(condition_img)
            condition_img = (255 - condition_img) / 255
            word_prompts = ["KDG", "科达股份", "证券代码：600986", "数字营销领军集团", " ", " ", " ", " "]
            test_list.append(([condition_img, hint, img], [0, 0], "keda group logo, that reads KDG , 科达股份 , 证券代码：600986 , 数字营销领军集团 ."))
            #
            hint = Image.open("assets/hint.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            img = Image.open("assets/hint_imgs.jpg").resize(
                (condition_size, condition_size)
            )
            condition_img = Image.open("assets/hint_imgs_word.png").resize(
                (condition_size, condition_size)
            ).convert('RGB')
            attnmask = None
            
            hint = np.array(hint) / 255
            # img = np.array(img)
            condition_img = np.array(condition_img)
            condition_img = (255 - condition_img) / 255
            word_prompts = ["LESOTHO", "COLLEGE OF", "RE BONA LESELI LESEL", "EDUCATION", " ", " ", " ", " "]
            # test_list.append(([condition_img, hint, img], [0, 0], "the chinese government has been accused of using the coronavirus outbreak to crack down on political dissidents, that reads 亵女下属，老鼎、丁 , 冬、悦尔、九戎、阿 , 甘知情却不处理，请 , 求公司还我公道！ ."))
            # test_list.append(([condition_img, hint, img], [0, 0], "a cartoon character with a red exclamation mark and chinese writing, that reads 对号入座, 公司倒闭 , 五个 , 迹象 ."))
            test_list.append(([condition_img, hint, img], [0, 0], "lepto college of education, the written materials on the picture: LESOTHO , COLLEGE OF , RE BONA LESELI LESEL , EDUCATION ."))
        else:
            raise NotImplementedError

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        for i, (condition_img, position_delta, prompt) in enumerate(test_list):
            if condition_type == 'word_fill':
                condition = Condition(
                        condition_type=condition_type,
                        condition=condition_img,
                        position_delta=position_delta,
                    )
                res = generate_fill(
                    pl_module.flux_pipe,
                    prompt=prompt,
                    conditions=[condition],
                    height=target_size,
                    width=target_size,
                    generator=generator,
                    model_config=pl_module.model_config,
                    default_lora=True,
                )
                res.images[0].save(
                    os.path.join(save_path, f"{file_name}_{condition_type}_{i}.jpg")
                )

