import os
import sys
import copy
import random
import numpy as np
import pickle
import torch
from torchvision import transforms
from torch.special import erf
from diffusers import DPMSolverMultistepScheduler
from scipy.special import betainc
import logging
from Crypto.Cipher import ChaCha20

logging.basicConfig(level=logging.ERROR)
from watermarkers.networks import BaseWatermarker

from .inverse_stable_diffusion import InversableStableDiffusionPipeline


class Gs(BaseWatermarker):
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        data_path="Gustavosta/Stable-Diffusion-Prompts",
        watermark_path="gs_watermark.pkl",
        watermark_size=4 * 64 * 64,
        batch_size=64,
        device="cuda",
    ):
        image_size = 512
        scheduler = DPMSolverMultistepScheduler.from_pretrained(checkpoint, subfolder='scheduler')
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            checkpoint,
            scheduler=scheduler,
            torch_dtype=torch.float16,
            revision='fp16',
        ).to(device)
        pipe.safety_checker = None
        pipe.set_progress_bar_config(disable=True)

        self.tform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )

        self.prompt = ""
        self.num_inference_steps = 50
        self.guidance_scale = 7.5
        self.ch = 1
        self.hw = 8
        self.user_number = 1000000
        self.fpr = 0.000001
        self.tau_bits = None
        self.tau_onebit = None

        marklength = watermark_size//(self.ch * self.hw * self.hw)
        for i in range(marklength):
            fpr_onebit = betainc(i+1, marklength-i, 0.5)
            fpr_bits = betainc(i+1, marklength-i, 0.5) * self.user_number
            if fpr_onebit <= self.fpr and self.tau_onebit is None:
                self.tau_onebit = i / marklength
            if fpr_bits <= self.fpr and self.tau_bits is None:
                self.tau_bits = i / marklength

        super().__init__(
            pipe,
            pipe,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=4 * 64 * 64,
            batch_size=batch_size,
            device=device,
        )

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]

    def init_encoder(self, encoder_checkpoint):
        return encoder_checkpoint

    def init_decoder(self, decoder_checkpoint):
        return decoder_checkpoint

    def init_watermark(self, watermark_path):
        with open(watermark_path, "rb") as f:
            watermark_m, self.key, self.nonce, self.watermark_raw = pickle.load(f)
        watermark_m = torch.from_numpy(watermark_m).to(dtype=torch.float16, device=self.device)
        self.watermark_raw = self.watermark_raw.to(device=self.device)
        return watermark_m

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        init_latents_w = torch.randn(
            (num_images, 4, 64, 64), dtype=torch.float16, device=self.device
        )
        return (init_latents_w, prompts, image_size)

    def post_process_raw(self, x): 
        return self.encoder(
            x[1],
            num_images_per_prompt=1,
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.num_inference_steps,
            height=x[2],
            width=x[2],
            latents=x[0],
        )

    def encode(self, x, with_grad=False):
        init_latents_w, prompts, orig_size = x
        encoded = []
        n_batch = int(np.ceil(len(init_latents_w) / self.batch_size))

        for step in range(n_batch):
            latents_i = init_latents_w[
                step * self.batch_size : (step + 1) * self.batch_size
            ].to(self.device)
            prompts_i = prompts[step * self.batch_size : (step + 1) * self.batch_size]
            imgs = (latents_i, prompts_i, orig_size)
            msg_batch = (
                self.watermark.repeat(latents_i.shape[0], 1)
                if self.watermark is not None
                else None
            )
            if not with_grad:
                with torch.no_grad():
                    encoded_image_batch = self._encode_batch(imgs, msg_batch)
            else:
                encoded_image_batch = self._encode_batch(imgs, msg_batch)
            encoded.append(encoded_image_batch)
        encoded = torch.concat(encoded).view(-1, 3, self.image_size, self.image_size)
        return transforms.Resize((orig_size, orig_size), antialias=None)(encoded).to(
            init_latents_w.device
        )

    def get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images = self.get_raw_images(input_dir, num_images, image_size=image_size)
        return self.post_process_raw(raw_images), self.encode(raw_images)

    def _encode_batch(self, x_batch, msg_batch):
        init_latents_w, prompts, image_size = x_batch
        mask = torch.where(self.watermark == 0, -1, self.watermark).to(device=self.device)
        new_latents = (
            (mask * torch.abs(init_latents_w.reshape(self.watermark.shape)))
            .reshape(init_latents_w.shape)
            .to(self.device)
        )
        return self.post_process_raw((new_latents, prompts, self.image_size))

    def _decode_batch_raw(self, x):
        text_embeddings = self.decoder.get_text_embedding(self.prompt).to(self.device)
        x = (2.0 * self.tform(x) - 1.0).to(text_embeddings.dtype)
        image_latents = self.decoder.get_image_latents(x, sample=False)
        reversed_latents = (
            self.decoder.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=1,
                num_inference_steps=self.num_inference_steps,
            )
            .to(torch.float16)
        )
        
        reversed_m = (reversed_latents > 0).int().flatten().cpu().numpy()
        cipher = ChaCha20.new(key=self.key, nonce=self.nonce)
        sd_byte = cipher.decrypt(np.packbits(reversed_m).tobytes())
        sd_bit = np.unpackbits(np.frombuffer(sd_byte, dtype=np.uint8))
        reversed_sd = torch.from_numpy(sd_bit).to(self.device).reshape(reversed_latents.shape).unsqueeze(0).to(torch.uint8)##

        threshold = self.ch * self.hw * self.hw // 2
        ch_stride = 4 // self.ch
        hw_stride = 64 // self.hw
        ch_list = [ch_stride] * self.ch
        hw_list = [hw_stride] * self.hw

        split_dim1 = torch.cat(torch.split(reversed_sd, tuple(ch_list), dim=2), dim=1)##
        split_dim2 = torch.cat(torch.split(split_dim1, tuple(hw_list), dim=3), dim=1)##
        split_dim3 = torch.cat(torch.split(split_dim2, tuple(hw_list), dim=4), dim=1)##

        vote = torch.sum(split_dim3, dim=1).clone()##
        vote[vote <= threshold] = 0
        vote[vote > threshold] = 1
        return vote.to(self.device)

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def stats(self, imgs, decoded, msg_batch): 
        correct = (decoded == self.watermark_raw).float().mean(dim=(1, 2, 3))
        return correct

    def threshold(self, n):####
        return self.tau_bits
