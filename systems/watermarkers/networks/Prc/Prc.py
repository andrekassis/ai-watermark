import os
import sys
import copy
import random
import numpy as np
from scipy.sparse import csr_matrix
import pickle
import torch
from torchvision import transforms
from torch.special import erf
from diffusers import DPMSolverMultistepScheduler
import logging
import galois
from scipy.special import binom

GF = galois.GF(2)

logging.basicConfig(level=logging.ERROR)
from watermarkers.networks import BaseWatermarker

from .inverse_stable_diffusion import InversableStableDiffusionPipeline


class Prc(BaseWatermarker):
    def __init__(
        self,
        checkpoint="stabilityai/stable-diffusion-2-1-base",
        data_path="Gustavosta/Stable-Diffusion-Prompts",
        watermark_path="prc_codeword.pkl",
        watermark_size=4 * 64 * 64,
        batch_size=64,
        device="cuda",
    ):
        image_size = 512
        hf_cache_dir = "cache/"
        scheduler = DPMSolverMultistepScheduler(
            beta_end=0.012,
            beta_schedule="scaled_linear",
            beta_start=0.00085,
            num_train_timesteps=1000,
            prediction_type="epsilon",
            steps_offset=1,
            trained_betas=None,
            solver_order=1,
        )
        pipe = InversableStableDiffusionPipeline.from_pretrained(
            checkpoint,
            scheduler=scheduler,
            torch_dtype=torch.float32,
            cache_dir=hf_cache_dir,
        ).to(device)
        pipe.set_progress_bar_config(disable=True)

        super().__init__(
            pipe,
            pipe,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_size,
            batch_size=batch_size,
            device=device,
        )

        self.tform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
            ]
        )

        self.decoder_inv = False
        self.prompt = ""
        self.test_num_inference_steps = 50
        self.guidance_scale = 3.0
        self.inv_order = 0
        self.variances = 1.5

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]

    def init_encoder(self, encoder_checkpoint):
        return encoder_checkpoint

    def init_decoder(self, decoder_checkpoint):
        return decoder_checkpoint

    def init_prc_keys(self, path):
        with open(path, "rb") as f:
            keys = pickle.load(f)
        return keys

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = random.sample(list(range(len(self.dataset))), num_images)
        prompts = [self.dataset[i]["Prompt"] for i in prompts]
        init_latents_w = torch.randn(
            (num_images, 4, 64, 64), dtype=torch.float64, device=self.device
        )
        return (init_latents_w, prompts, image_size)

    def post_process_raw(self, x):
        return self.encoder(
            x[1],
            guidance_scale=self.guidance_scale,
            num_inference_steps=self.test_num_inference_steps,
            height=x[2],
            width=x[2],
            latents=x[0],
        ).view(-1, 3, x[2], x[2])

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

    def _encode_batch(self, x_batch, msg_batch):
        init_latents_w, prompts, image_size = x_batch
        return self.post_process_raw(
            (
                torch.abs(init_latents_w) * msg_batch.view(init_latents_w.shape),
                prompts,
                self.image_size,
            )
        )

    def KeyGen(
        self,
        n,
        message_length=512,
        false_positive_rate=1e-9,
        t=3,
        g=None,
        r=None,
        noise_rate=None,
    ):
        num_test_bits = int(np.ceil(np.log2(1 / false_positive_rate)))
        secpar = int(np.log2(binom(n, t)))
        if g is None:
            g = secpar
        if noise_rate is None:
            noise_rate = 1 - 2 ** (-secpar / g**2)
        k = message_length + g + num_test_bits
        if r is None:
            r = n - k - secpar

        generator_matrix = GF.Random((n, k))
        row_indices = []
        col_indices = []
        data = []
        for row in range(r):
            chosen_indices = np.random.choice(n - r + row, t - 1, replace=False)
            chosen_indices = np.append(chosen_indices, n - r + row)
            row_indices.extend([row] * t)
            col_indices.extend(chosen_indices)
            data.extend([1] * t)
            generator_matrix[n - r + row] = generator_matrix[chosen_indices[:-1]].sum(
                axis=0
            )
        parity_check_matrix = csr_matrix((data, (row_indices, col_indices)))

        max_bp_iter = int(np.log(n) / np.log(t))

        one_time_pad = GF.Random(n)
        test_bits = GF.Random(num_test_bits)

        permutation = np.random.permutation(n)
        generator_matrix = generator_matrix[permutation]
        one_time_pad = one_time_pad[permutation]
        parity_check_matrix = parity_check_matrix[:, permutation]
        encoding_key = (generator_matrix, one_time_pad, test_bits, g, noise_rate)
        decoding_key = (
            generator_matrix,
            parity_check_matrix,
            one_time_pad,
            false_positive_rate,
            noise_rate,
            test_bits,
            g,
            max_bp_iter,
            t,
        )
        return encoding_key, decoding_key

    def Encode(self, encoding_key, message=None):
        generator_matrix, one_time_pad, test_bits, g, noise_rate = encoding_key
        n, k = generator_matrix.shape

        if message is None:
            payload = np.concatenate((test_bits, GF.Random(k - len(test_bits))))
        else:
            assert len(message) <= k - len(test_bits) - g, "Message is too long"
            payload = np.concatenate(
                (
                    test_bits,
                    GF.Random(g),
                    GF(message),
                    GF.Zeros(k - len(test_bits) - g - len(message)),
                )
            )

        error = GF(np.random.binomial(1, noise_rate, n))

        return 1 - 2 * torch.tensor(
            payload @ generator_matrix.T + one_time_pad + error, dtype=float
        )

    def init_watermark(self, watermark_path):
        if not os.path.exists(watermark_path):
            encoding_key, decoding_key = self.KeyGen(
                self.watermark_length, false_positive_rate=0.00001, t=3
            )
            watermark = self.Encode(encoding_key).view(1, -1).detach().cpu().numpy()
            base = "/".join(watermark_path.split("/")[:-1])
            if base != "":
                os.makedirs(base, exist_ok=True)
            with open(watermark_path, "wb") as f:
                data = (watermark, decoding_key)
                pickle.dump(data, f)
        else:
            with open(watermark_path, "rb") as f:
                watermark, decoding_key = pickle.load(f)

        _, parity_check_matrix, one_time_pad, self.fpr, self.noise_rate, _, _, _, t = (
            decoding_key
        )
        self.one_time_pad = torch.from_numpy(np.array(one_time_pad, dtype=float)).to(
            self.device
        )
        self.parity_indices = torch.from_numpy(
            parity_check_matrix.indices.reshape(parity_check_matrix.shape[0], t)
        ).to(self.device)

        return (
            torch.from_numpy(watermark)
            .to(dtype=torch.float64, device=self.device)
            .view(1, -1)
        )

    def _decode_batch_raw(self, x):
        x = 2.0 * self.tform(x) - 1.0
        image_latents = (
            self.decoder.decoder_inv(x)
            if self.decoder_inv
            else self.decoder.get_image_latents(x, sample=False)
        )
        text_embeddings_tuple = self.decoder.encode_prompt(
            self.prompt, x.device, 1, self.guidance_scale > 1.0, None
        )
        text_embeddings = torch.cat(
            [text_embeddings_tuple[1], text_embeddings_tuple[0]]
        )
        reversed_latents = (
            self.decoder.forward_diffusion(
                latents=image_latents,
                text_embeddings=text_embeddings,
                guidance_scale=self.guidance_scale,
                num_inference_steps=self.test_num_inference_steps,
                inverse_opt=(self.inv_order != 0),
                inv_order=self.inv_order,
            )
            .to(torch.float64)
            .flatten(start_dim=1)
        )
        return erf(
            reversed_latents / np.sqrt(2 * self.variances * (1 + self.variances))
        )

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def stats(self, imgs, decoded, msg_batch):
        posteriors = (
            (1 - 2 * self.noise_rate)
            * (1 - 2 * self.one_time_pad.unsqueeze(0))
            * decoded
        )
        Pi = torch.prod(posteriors[:, self.parity_indices], dim=-1)
        log_plus = torch.log((1 + Pi) / 2)
        log_minus = torch.log((1 - Pi) / 2)
        log_prod = log_plus + log_minus
        const = 0.5 * torch.sum(
            log_plus.square() + log_minus.square() - 0.5 * log_prod.square(), dim=-1
        )
        threshold = torch.sqrt(2 * const * np.log(1 / self.fpr)) + 0.5 * log_prod.sum(
            dim=-1
        )
        return (log_plus.sum(dim=-1) - threshold).double()

    def threshold(self, n):
        return 0
