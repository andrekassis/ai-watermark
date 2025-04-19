import torch
from torchvision import transforms
from PIL import Image
from watermarkers.networks import RandomWatermarkers
from .model import Encoder, Extractor


class InvisMark(RandomWatermarkers):
    def __init__(
        self,
        weights_path="weights.ckpt",
        watermark_path="invis_watermark.npy",
        watermark_length=100,
        image_size=256,
        batch_size=64,
        device="cuda",
    ):
 
        self.state_dict = torch.load(weights_path)
        self.tform = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

        super().__init__(
            weights_path,
            weights_path,
            watermark_path=watermark_path,
            image_size=image_size,
            watermark_length=watermark_length,
            batch_size=batch_size,
            device=device,
        )

    def init_encoder(self, encoder_checkpoint):
        encoder = Encoder().to(self.device)
        encoder.load_state_dict(self.state_dict['encoder_state_dict'])
        encoder.train()
        return encoder

    def init_decoder(self, decoder_checkpoint):
        decoder = Extractor().to(self.device)
        decoder.load_state_dict(self.state_dict['decoder_state_dict'])
        decoder.train()
        return decoder

    def _encode_batch(self, x_batch, msg_batch):
        images = self.tform(x_batch) 
        secret =  torch.cat([self.watermark] * x_batch.shape[0], dim=0)         
        encoded_output = self.encoder(images, secret)
        output = torch.clamp(encoded_output, min=-1.0, max=1.0).to(self.device)
        output = output * 0.5 + 0.5
        return output.view(-1, 3, self.image_size, self.image_size)

    def _decode_batch_raw(self, x):
        images = self.tform(x)
        decoded = self.decoder(images)
        return (decoded >= 0.5).float()

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

