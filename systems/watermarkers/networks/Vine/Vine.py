import random
import os
import pickle
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from watermarkers.networks import BaseWatermarker
from .vine_turbo import VINE_Turbo
from .stega_encoder_decoder import CustomConvNeXt


class Vine(BaseWatermarker):
    def __init__(
        self,
        encoder_path="Shilin-LU/VINE-R-Enc",
        decoder_path="Shilin-LU/VINE-R-Dec",
        data_path="datasets/coco/val_class", 
        watermark_path="vine_watermark.pkl",
        watermark_length=100,
        batch_size=64,
        device="cuda",
    ):
        image_size = 512

        super().__init__(
            encoder_path,
            decoder_path,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )     

        self.t_val_256 = transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC), 
            ])
        self.t_val_512 = transforms.Compose([
                transforms.Resize(512, interpolation=transforms.InterpolationMode.BICUBIC), 
            ])

        self.data_path = data_path
        self.dataset =  [f for f in os.listdir(data_path)]

    def init_encoder(self, encoder_checkpoint):
        encoder = VINE_Turbo.from_pretrained(encoder_checkpoint)
        encoder.to(self.device)
        return encoder

    def init_decoder(self, decoder_checkpoint):
        decoder = CustomConvNeXt.from_pretrained(decoder_checkpoint)
        decoder.to(self.device)
        return decoder

    def init_watermark(self, watermark_path):
        with open(watermark_path, "rb") as f:
            watermark = pickle.load(f)
        return watermark.to(self.device)

    def get_secrets(self, input_dir, num_images, image_size=256):
        return super().get_raw_images(input_dir, num_images, image_size)

    def crop_to_square(self, image):
        width, height = image.size

        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def get_raw_images(self, input_dir, num_images, image_size=256):  
        selected_images = random.sample(self.dataset, num_images)
        image_list = []
        for img_name in selected_images:
            img_path = os.path.join(self.data_path, img_name)
            image = Image.open(img_path).convert('RGB')
            if image.size[0] != image.size[1]:
                image = self.crop_to_square(image)
            image = transforms.ToTensor()(self.t_val_512(image)).to(self.device)
            image_list.append(image)
        image_list = torch.stack(image_list)
        return image_list, image_size

    def post_process_raw(self, x):
        images, image_size = x
        return images

    def encode(self, x, with_grad=False):
        samples, orig_size = x
        encoded = []
        n_batch = int(np.ceil(len(samples) / self.batch_size))

        for step in range(n_batch):
            img_samples = samples[step * self.batch_size : (step + 1) * self.batch_size]
            imgs = (img_samples, orig_size)
            msg_batch = (
                self.watermark.repeat(len(imgs), 1)
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
            self.device
        )

    def get_watermarked_images(self, input_dir, num_images, image_size=256): 
        raw_images = self.get_raw_images(input_dir, num_images, image_size=image_size)
        return self.post_process_raw(raw_images), self.encode(raw_images)

    def _encode_batch(self, x_batch, msg_batch):
        images, image_size = x_batch
        resized_img = self.t_val_256(images)
        resized_img = 2.0 * resized_img - 1.0
        input_image = 2.0 * images - 1.0

        encoded_image_256 = self.encoder(resized_img, self.watermark)
        residual_256 = encoded_image_256 - resized_img 
        residual_512 = self.t_val_512(residual_256) 
        encoded_image = residual_512 + input_image
        encoded_image = encoded_image * 0.5 + 0.5
        encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

        output_pil = encoded_image.view(-1, 3, self.image_size, self.image_size)
        return self.post_process_raw((output_pil, image_size))

    def _decode_batch_raw(self, x):
        images = self.t_val_256(x)
        pred_watermark = self.decoder(images)
        pred_watermark = torch.round(pred_watermark)
        return pred_watermark

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def stats(self, imgs, decoded, msg_batch): 
        correct = (decoded == self.watermark).float().mean(dim=(1))
        return correct

    def threshold(self, n):
        return 0.75
