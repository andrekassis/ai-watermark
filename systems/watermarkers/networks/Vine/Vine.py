import torch
from torchvision import transforms
from PIL import Image
from watermarkers.networks import RandomWatermarkers
from .vine_turbo import VINE_Turbo
from .stega_encoder_decoder import CustomConvNeXt


class Vine(RandomWatermarkers):
    def __init__(
        self,
        encoder_path="Shilin-LU/VINE-R-Enc",
        decoder_path="Shilin-LU/VINE-R-Dec",
        watermark_path="vine_watermark.pkl",
        watermark_length=100,
        image_size=512,
        batch_size=64,
        device="cuda",
    ):
        super().__init__(
            encoder_path,
            decoder_path,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )

        self.t_val_256 = transforms.Resize(
            image_size // 2, interpolation=transforms.InterpolationMode.BICUBIC
        )
        self.t_val_512 = transforms.Resize(
            image_size, interpolation=transforms.InterpolationMode.BICUBIC
        )

    def init_encoder(self, encoder_checkpoint):
        encoder = VINE_Turbo.from_pretrained(encoder_checkpoint)
        encoder.to(self.device)
        return encoder

    def init_decoder(self, decoder_checkpoint):
        decoder = CustomConvNeXt.from_pretrained(decoder_checkpoint)
        decoder.to(self.device)
        return decoder

    def crop_to_square(self, image):
        width, height = image.size
        if height == width:
            return image

        min_side = min(width, height)
        left = (width - min_side) // 2
        top = (height - min_side) // 2
        right = left + min_side
        bottom = top + min_side

        cropped_image = image.crop((left, top, right, bottom))
        return cropped_image

    def load(self, path, image_size=256):
        return (
            transforms.Resize(
                image_size, interpolation=transforms.InterpolationMode.BICUBIC
            )(
                transforms.ToTensor()(
                    self.crop_to_square(Image.open(path).convert("RGB"))
                )
            )
            .view(1, 3, image_size, image_size)
            .to(self.device)
        )

    def _encode_batch(self, x_batch, msg_batch):
        images = x_batch
        resized_img = self.t_val_256(images)
        resized_img = 2.0 * resized_img - 1.0
        input_image = 2.0 * images - 1.0

        encoded_image_256 = self.encoder(resized_img, self.watermark)
        residual_256 = encoded_image_256 - resized_img
        residual_512 = self.t_val_512(residual_256)
        encoded_image = residual_512 + input_image
        encoded_image = encoded_image * 0.5 + 0.5
        encoded_image = torch.clamp(encoded_image, min=0.0, max=1.0)

        return encoded_image.view(-1, 3, self.image_size, self.image_size)

    def _decode_batch_raw(self, x):
        return torch.round(self.decoder(self.t_val_256(x)))

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)
