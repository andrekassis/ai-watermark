import os
import random
import torch
import numpy as np
from abc import ABC
from torchvision import transforms
from PIL import Image
from math import comb


class BaseWatermarker(ABC):
    def __init__(
        self,
        encoder_checkpoint,
        decoder_checkpoint,
        watermark_path,
        watermark_length,
        image_size,
        batch_size,
        device,
    ):
        super().__init__()
        self.watermark_length = watermark_length
        self.image_size = image_size
        self.resizer = transforms.Resize((image_size, image_size), antialias=None)
        self.batch_size = batch_size
        self.device = device

        self.decoder = self.init_decoder(decoder_checkpoint)
        self.encoder = self.init_encoder(encoder_checkpoint)
        self.watermark = self.init_watermark(watermark_path)
        self.watermark = (
            self.watermark.to(self.device) if self.watermark is not None else None
        )
        self.acceptance_thresh = self.threshold(self.watermark_length)
        ###
        self.iters = 0
        ###

    def threshold(self, n):
        if n is None:
            return None
        event = lambda k: np.array(
            [comb(n, j) * (0.5**n) for j in range(k, n + 1)]
        ).sum()
        for i in range(1, n + 1):
            ret = event(i)
            if ret < 0.01:
                return round(i / float(n), 2)

    def init_watermark(self, watermark_path):
        pass

    def init_decoder(self, decoder_checkpoint):
        pass

    def init_encoder(self, encoder_checkpoint):
        pass

    def _encode_batch(self, x_batch, msg_batch):
        pass

    def _decode_batch(self, x_batch, msg_batch):
        pass

    def is_detected(self, accs):
        return accs >= self.acceptance_thresh

    def _decode_batch_raw(self, x):
        pass

    def err(self, x_batch, msg_batch):
        return torch.abs(
            x_batch.view(x_batch.shape[0], -1) - msg_batch.view(msg_batch.shape[0], -1)
        ).mean(-1)

    def process_encoder_input(self, x):
        return self.resizer(x)

    def encode(self, x, with_grad=False):
        orig_size = x.shape[-1] if len(x.shape) > 1 else None
        encoded = []
        n_batch = int(np.ceil(len(x) / self.batch_size))
        for step in range(n_batch):
            imgs = self.process_encoder_input(
                x[step * self.batch_size : (step + 1) * self.batch_size].to(self.device)
            )
            msg_batch = (
                self.watermark.repeat(imgs.shape[0], 1)
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
        if orig_size is not None:
            encoded = transforms.Resize((orig_size, orig_size), antialias=None)(
                encoded
            ).to(self.device)
        return encoded

    def decode_batch_raw(self, x):
        x = self.resizer(x.to(self.device))
        return self._decode_batch_raw(x)

    def __call__(self, x):
        accs = np.zeros((0,), dtype=np.float32)
        n_batch = int(np.ceil(len(x) / self.batch_size))
        for step in range(n_batch):
            imgs = self.resizer(
                x[step * self.batch_size : (step + 1) * self.batch_size].to(self.device)
            )
            msg_batch = (
                self.watermark.repeat(imgs.shape[0], 1)
                if self.watermark is not None
                else None
            )
            with torch.no_grad():
                decoded = self._decode_batch(imgs, msg_batch)
            accs = np.concatenate(
                (
                    accs,
                    self.stats(imgs, decoded, msg_batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .round(2),
                )
            )
        return accs

    def evaluate(self, x):
        accs = np.zeros((0,), dtype=np.float32)
        n_batch = int(np.ceil(len(x) / self.batch_size))
        dec = []
        for step in range(n_batch):
            imgs = self.resizer(
                x[step * self.batch_size : (step + 1) * self.batch_size].to(self.device)
            )
            msg_batch = (
                self.watermark.repeat(imgs.shape[0], 1)
                if self.watermark is not None
                else None
            )
            decoded = self._decode_batch_eval_mode(imgs, msg_batch)
            accs = np.concatenate(
                (
                    accs,
                    self.stats(imgs, decoded, msg_batch)
                    .detach()
                    .cpu()
                    .numpy()
                    .round(2),
                )
            )
            dec.extend([d.detach() for d in decoded])
        return accs, torch.concat(dec)

    def _decode_batch_eval_mode(self, imgs, msg_batch):
        return self._decode_batch(imgs, msg_batch)

    def eval_batch(self, x, w=None):
        x = self.resizer(x.to(self.device))
        msg_batch = (
            (
                self.watermark.repeat(x.shape[0], 1)
                if self.watermark is not None
                else None
            )
            if w is None
            else w
        )
        decoded = self._decode_batch(x, msg_batch)
        acc = self.stats(x, decoded, msg_batch)
        return acc, self.is_detected(acc)

    def stats(self, imgs, decoded, msg_batch):
        return 1 - self.err(decoded, msg_batch)

    def save(self, image, path):
        transforms.ToPILImage()(
            self.resizer(image.view(1, 3, image.shape[-1], image.shape[-1]))
            .detach()
            .cpu()
            .squeeze()
        ).save(path)

    def load(self, path, image_size=256):
        return (
            transforms.Resize((image_size, image_size), antialias=None)(
                transforms.ToTensor()(Image.open(path).convert("RGB"))
            )
            .view(1, 3, image_size, image_size)
            .to(self.device)
        )

    def post_process_raw(self, x):
        return x

    def get_raw_images(self, input_dir, num_images, image_size=256):
        if input_dir is None:
            return torch.tensor([0] * num_images)

        file_list = [os.path.join(input_dir, f) for f in os.listdir(input_dir)][
            self.iters * num_images : (self.iters + 1) * num_images
        ]
        self.iters += 1

        return torch.concat([self.load(f, image_size=image_size) for f in file_list])

    def get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images = self.get_raw_images(input_dir, num_images, image_size=image_size)
        return self.post_process_raw(raw_images), self.encode(raw_images)

    def get_unmarked_images(self, x, input_dir, num_images, image_size=256):
        return x

    def get_secrets(self, input_dir, num_images, image_size=256):
        return self.get_raw_images(input_dir, num_images, image_size)


class RandomWatermarkers(BaseWatermarker):
    def init_watermark(self, watermark_path):
        if os.path.exists(watermark_path):
            msg = np.load(watermark_path)[: self.watermark_length]
        else:
            msg = np.random.choice([0, 1], (1, self.watermark_length))
            base = "/".join(watermark_path.split("/")[:-1])
            if base != "":
                os.makedirs(base, exist_ok=True)
            np.save(watermark_path, msg)
        return torch.Tensor(msg).float()


class FixedWatermarkers(BaseWatermarker):
    def init_watermark(self, watermark_path):
        assert os.path.exists(watermark_path)
        return torch.tensor(np.load(watermark_path)).float().to(self.device).view(1, -1)
