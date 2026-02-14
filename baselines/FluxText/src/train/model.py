
from diffusers.pipelines import FluxFillPipeline
import lightning as L
from peft import LoraConfig, get_peft_model_state_dict
import prodigyopt
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from ..flux.transformer import tranformer_forward
from ..flux.condition import Condition
from ..flux.pipeline_tools import encode_images, prepare_text_input
from ..loss.ocr_loss.odm_loss import ODMLoss
from ..loss.ocr_loss.ocr_loss import OCRLoss
from ..text_encoder.byt5_encoder import GlyphByt5Encoder


class OminiModelFIll(L.LightningModule):
    def __init__(
        self,
        flux_pipe_id: str,
        lora_path: str = None,
        reuse_lora_path: str = None,
        lora_config: dict = None,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        model_config: dict = {},
        optimizer_config: dict = None,
        gradient_checkpointing: bool = False,
        odm_loss_config: dict = None,
        ocr_loss_config: dict = None,
        byt5_encoder_config: dict = None,
    ):
        # Initialize the LightningModule
        super().__init__()
        self.model_config = model_config
        self.optimizer_config = optimizer_config

        # Load the Flux pipeline
        self.flux_pipe: FluxFillPipeline = (
            FluxFillPipeline.from_pretrained(flux_pipe_id).to(dtype=dtype).to(device)
        )
        self.transformer = self.flux_pipe.transformer
        self.transformer.gradient_checkpointing = gradient_checkpointing
        self.transformer.train()

        # Freeze the Flux pipeline
        self.flux_pipe.text_encoder.requires_grad_(False).eval()
        self.flux_pipe.text_encoder_2.requires_grad_(False).eval()
        self.flux_pipe.vae.requires_grad_(False).eval()

        self.vae_scale_factor = 8

        # Initialize LoRA layers
        self.lora_layers = self.init_lora(lora_path, lora_config)
        # reuse the weight
        if reuse_lora_path is not None:
            print(f"reuse the lora path: {reuse_lora_path}")
            state_dict = load_file(reuse_lora_path)
            state_dict1 = {x.replace('lora_A', 'lora_A.default').replace('lora_B', 'lora_B.default').replace('transformer.', ''): v for x, v in state_dict.items()}
            self.transformer.load_state_dict(state_dict1, strict=False)

        # Initialize ODM layers
        if odm_loss_config is not None:
            self.odm_loss = ODMLoss(**odm_loss_config)
        else:
            self.odm_loss = None

        # Initialize ocr loss
        if ocr_loss_config is not None:
            self.ocr_loss = OCRLoss(device=device, dtype=dtype, **ocr_loss_config)
        else:
            self.ocr_loss = None

        self.to(device).to(dtype)

    def init_lora(self, lora_path: str, lora_config: dict):
        assert lora_path or lora_config
        if lora_path:
            # TODO: Implement this
            raise NotImplementedError
        else:
            self.transformer.add_adapter(LoraConfig(**lora_config))
            # TODO: Check if this is correct (p.requires_grad)
            lora_layers = filter(
                lambda p: p.requires_grad, self.transformer.parameters()
            )
        return list(lora_layers)

    def save_lora(self, path: str):
        FluxFillPipeline.save_lora_weights(
            save_directory=path,
            transformer_lora_layers=get_peft_model_state_dict(self.transformer),
            safe_serialization=True,
        )

    def configure_optimizers(self):
        # Freeze the transformer
        self.transformer.requires_grad_(False)
        opt_config = self.optimizer_config

        # Set the trainable parameters
        self.trainable_params = self.lora_layers

        # Unfreeze trainable parameters
        for p in self.trainable_params:
            p.requires_grad_(True)

        # Initialize the optimizer
        if opt_config["type"] == "AdamW":
            optimizer = torch.optim.AdamW(self.trainable_params, **opt_config["params"])
        elif opt_config["type"] == "Prodigy":
            optimizer = prodigyopt.Prodigy(
                self.trainable_params,
                **opt_config["params"],
            )
        elif opt_config["type"] == "SGD":
            optimizer = torch.optim.SGD(self.trainable_params, **opt_config["params"])
        else:
            raise NotImplementedError

        return optimizer

    def training_step(self, batch, batch_idx):
        step_loss = self.step(batch)
        self.log_loss = (
            step_loss.item()
            if not hasattr(self, "log_loss")
            else self.log_loss * 0.95 + step_loss.item() * 0.05
        )
        return step_loss

    def step(self, batch):
        imgs = batch["image"]
        ref_imgs = batch["ref_image"]
        conditions = batch["condition"]
        hints = batch["hint"]
        condition_types = batch["condition_type"]
        prompts = batch["description"]
        position_delta = batch["position_delta"][0]

        mask_image = hints[:, 0, :, :]

        ori_height = imgs.shape[2]
        ori_width = imgs.shape[3]
        height = 2 * (int(ori_height) // (self.vae_scale_factor * 2))
        width = 2 * (int(ori_width) // (self.vae_scale_factor * 2))
        condition_height = conditions.shape[2]
        condition_width = conditions.shape[3]
        condition_height = 2 * (int(condition_height) // (self.vae_scale_factor * 2))
        condition_width = 2 * (int(condition_width) // (self.vae_scale_factor * 2))
        batch_size = imgs.shape[0]

        # Prepare inputs
        with torch.no_grad():
            # Prepare image input
            x_0, img_ids = encode_images(self.flux_pipe, imgs)
            x_0_ref, _ = encode_images(self.flux_pipe, ref_imgs)

            # Prepare text input
            prompt_embeds, pooled_prompt_embeds, text_ids = prepare_text_input(
                self.flux_pipe, prompts
            )

            # Prepare t and x_t
            t = torch.sigmoid(torch.randn((imgs.shape[0],), device=self.device))
            x_1 = torch.randn_like(x_0).to(self.device)
            t_ = t.unsqueeze(1).unsqueeze(1)
            # x_t = ((1 - t_) * x_0 + t_ * x_1).to(self.dtype)
            x_t = ((1 - t_) * x_0_ref + t_ * x_1).to(self.dtype)

            # Prepare conditions
            condition_latents, condition_ids = encode_images(self.flux_pipe, conditions)

            # masked_image = imgs * (1 - mask_image)
            # masked_image_latents, _ = encode_images(self.flux_pipe, masked_image)
            imgs = self.flux_pipe.image_processor.preprocess(imgs, height=imgs.shape[2], width=imgs.shape[3]) 
            ref_imgs = self.flux_pipe.image_processor.preprocess(ref_imgs, height=imgs.shape[2], width=imgs.shape[3])
            mask_image = self.flux_pipe.mask_processor.preprocess(mask_image, height=imgs.shape[2], width=imgs.shape[3])
            # masked_image = imgs * (1 - mask_image)
            masked_image = ref_imgs * (1 - mask_image)
            def _encode_mask(images):
                images = images.to(self.flux_pipe.device).to(self.flux_pipe.dtype)
                images = self.flux_pipe.vae.encode(images).latent_dist.sample()
                images = (
                    images - self.flux_pipe.vae.config.shift_factor
                ) * self.flux_pipe.vae.config.scaling_factor
                condition_images = F.interpolate(images, size=(condition_height, condition_width), mode='nearest')

                images_tokens = self.flux_pipe._pack_latents(images, *images.shape)
                images_ids = self.flux_pipe._prepare_latent_image_ids(
                    images.shape[0],
                    images.shape[2],
                    images.shape[3],
                    self.flux_pipe.device,
                    self.flux_pipe.dtype,
                )
                conditon_images_tokens = self.flux_pipe._pack_latents(condition_images, *condition_images.shape)
                return images_tokens, images_ids, conditon_images_tokens
                # return images_tokens, images_ids
            masked_image_latents, _, conditon_masked_images_latents = _encode_mask(masked_image)
            # masked_image_latents, _ = _encode_mask(masked_image)

            # mask = mask_image * (1 - conditions[:, :1])
            mask = mask_image
            mask = mask.view(batch_size, height, self.vae_scale_factor, width, self.vae_scale_factor)
            mask = mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            mask = mask.reshape(
                batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width
            )  # batch_size, 8*8, height, width
            mask = self.flux_pipe._pack_latents(mask, batch_size, self.vae_scale_factor * self.vae_scale_factor, height, width)

            masked_image_latents = torch.cat((masked_image_latents, mask), dim=-1).to(self.dtype)

            condition_mask = mask_image
            condition_mask = F.interpolate(condition_mask, size=(conditions.shape[2], conditions.shape[3]), mode='nearest')
            condition_mask = condition_mask.view(batch_size, condition_height, self.vae_scale_factor, condition_width, self.vae_scale_factor)
            condition_mask = condition_mask.permute(0, 2, 4, 1, 3)  # batch_size, 8, 8, height, width
            condition_mask = condition_mask.reshape(
                batch_size, self.vae_scale_factor * self.vae_scale_factor, condition_height, condition_width
            )  # batch_size, 8*8, height, width
            condition_mask = self.flux_pipe._pack_latents(condition_mask, batch_size, self.vae_scale_factor * self.vae_scale_factor, condition_height, condition_width)

            conditon_masked_images_latents = torch.cat((conditon_masked_images_latents, condition_mask), dim=-1).to(self.dtype)

            # Add position delta
            condition_ids[:, 1] += position_delta[0]
            condition_ids[:, 2] += position_delta[1]

            # Prepare condition type
            condition_type_ids = torch.tensor(
                [
                    Condition.get_type_id(condition_type)
                    for condition_type in condition_types
                ]
            ).to(self.device)
            condition_type_ids = (
                torch.ones_like(condition_ids[:, 0]) * condition_type_ids[0]
            ).unsqueeze(1)

            # Prepare guidance
            guidance = (
                torch.ones_like(t).to(self.device)
                if self.transformer.config.guidance_embeds
                else None
            )

        # Forward pass
        transformer_out = tranformer_forward(
            self.transformer,
            # Model config
            model_config=self.model_config,
            # Inputs of the condition (new feature)
            condition_latents=torch.cat((condition_latents, conditon_masked_images_latents), dim=2),
            condition_ids=condition_ids,
            condition_type_ids=condition_type_ids,
            # Inputs to the original transformer
            hidden_states=torch.cat((x_t, masked_image_latents), dim=2),
            timestep=t,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=img_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )
        pred = transformer_out[0]

        res = {}
        # Compute loss
        loss = torch.nn.functional.mse_loss(pred, (x_1 - x_0), reduction="mean")
        res['loss_sd'] = loss

        loss_mask = F.interpolate(mask_image, scale_factor=0.0625, mode='nearest')
        loss_mask = loss_mask.reshape(batch_size, -1)
        loss_mask = loss_mask[:, :, None]
        loss_mask = torch.nn.functional.mse_loss(pred * loss_mask, (x_1 - x_0) * loss_mask, reduction="mean")
        loss = loss + loss_mask
        res['loss_mask'] = loss_mask

        # ODM loss
        if self.odm_loss is not None:
            latents = self.flux_pipe._unpack_latents(x_1-pred, ori_height, ori_width, self.vae_scale_factor)
            latents = (
                latents / self.flux_pipe.vae.config.scaling_factor
            ) + self.flux_pipe.vae.config.shift_factor
            image_pred = self.flux_pipe.vae.decode(latents, return_dict=False)[0]

            # t1 = self.odm_loss.loss(image_pred, imgs) * 0.01
            odm_loss = self.odm_loss.loss(image_pred, imgs, mask_image)
            loss = loss + odm_loss
            res['loss_odm'] = odm_loss
        else:
            image_pred = None

        # ocr loss
        if self.ocr_loss is not None:
            if image_pred is None:
                latents = self.flux_pipe._unpack_latents(x_1-pred, ori_height, ori_width, self.vae_scale_factor)
                latents = (
                    latents / self.flux_pipe.vae.config.scaling_factor
                ) + self.flux_pipe.vae.config.shift_factor
                image_pred = self.flux_pipe.vae.decode(latents, return_dict=False)[0]

            ocr_loss = self.ocr_loss.loss(image_pred, imgs, batch)
            res['loss_ocr'] = ocr_loss['loss_ocr']
            res['loss_ctc'] = ocr_loss['loss_ctc']
            loss = loss + ocr_loss['loss_ocr'] + ocr_loss['loss_ctc']

        res['loss'] = loss
        self.res = res

        self.last_t = t.mean().item()
        return loss
