import sys
import io
import random
import time
import logging
import pickle
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from google.api_core import exceptions
import vertexai
from vertexai.preview.vision_models import (
    ImageGenerationModel,
    WatermarkVerificationModel,
)
from vertexai.preview.vision_models import Image as VImage

from watermarkers.networks import BaseWatermarker


class SynthID(BaseWatermarker):
    def __init__(
        self,
        project_id,
        generation_model="imagen-3.0-generate-002",
        verification_model="imageverification@001",
        service="us-central1",
        data_path="datasets/treering_prompts.obj",
        image_size=1024,
        batch_size=64,
        device="cuda",
    ):
        vertexai.init(project=project_id, location=service)
        assert image_size <= 1024

        super().__init__(
            generation_model,
            verification_model,
            None,
            None,
            image_size,
            batch_size,
            device,
        )

        with open(data_path, "rb") as f:
            self.dataset = pickle.load(f)["test"]

        self.gen_kwargs = {
            "language": "en",
            "aspect_ratio": "1:1",
            "safety_filter_level": "block_some",
            "person_generation": "allow_adult",
            "number_of_images": 1,
        }
        self.acceptance_thresh = 0.0

        self.idcs = random.sample(list(range(len(self.dataset))), len(self.dataset))
        self.api_trials = 20
        self.trial_interval = 5

    def to_vimage(self, images):
        images = [transforms.ToPILImage()(image) for image in images]
        vout = []
        for image in images:
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            vout.append(VImage(img_byte_arr.getvalue()))
        return vout

    def init_watermark(self, watermark_path):
        return None

    def init_encoder(self, encoder_checkpoint):
        return ImageGenerationModel.from_pretrained(encoder_checkpoint)

    def init_decoder(self, decoder_checkpoint):
        return WatermarkVerificationModel.from_pretrained(decoder_checkpoint)

    def do_gen(self, prompt, add_watermark=False):
        trials = 0
        while True:
            try:
                response = self.encoder.generate_images(
                    prompt=prompt,
                    add_watermark=add_watermark,
                    **self.gen_kwargs,
                )
                break
            except exceptions.ResourceExhausted as e:
                trials += 1
                if trials == self.api_trials:
                    logging.error(
                        "Google's RPC seems to be unresponsive. Aborting now. Please rerun the "
                        "attack with a different seed to continue with a fresh set of samples."
                    )
                    sys.exit(1)
                time.sleep(self.trial_interval)
                continue
            except Exception as e:
                break
        try:
            return [response[0]]
        except:
            return []

    def gen_from_prompts(self, prompts, add_watermark=False):
        n_batch = int(np.ceil(len(prompts) / self.batch_size))
        samples = torch.empty(
            (0, 3, 1024, 1024), device=self.device, dtype=torch.float32
        )
        pout = []
        for step in range(n_batch):
            prompts_i = prompts[step * self.batch_size : (step + 1) * self.batch_size]
            x_samples = [
                [
                    prompt,
                    self.do_gen(
                        prompt=prompt,
                        add_watermark=add_watermark,
                    ),
                ]
                for prompt in prompts_i
            ]
            prompts_i = [x[0] for x in x_samples if len(x[1]) > 0]
            x_samples = [
                transforms.ToTensor()(Image.open(io.BytesIO(x[1][0]._image_bytes)))
                .unsqueeze(0)
                .to(self.device)
                for x in x_samples
                if len(x[1]) > 0
            ]
            x_samples = (
                torch.cat(x_samples).view(-1, 3, 1024, 1024)
                if len(x_samples) > 0
                else torch.empty(
                    (0, 3, 1024, 1024), device=self.device, dtype=torch.float32
                )
            )
            samples = torch.cat((samples, x_samples)).view(-1, 3, 1024, 1024)
            pout.extend(prompts_i)
        return (
            transforms.Resize((self.image_size, self.image_size), antialias=None)(
                samples
            ),
            pout,
        )

    def try_get_raw_images(self, num_images):
        prompts = [self.dataset[i]["Prompt"] for i in self.idcs[:num_images]]
        self.idcs = self.idcs[num_images:]
        return self.gen_from_prompts(prompts, add_watermark=False)

    def get_raw_images(self, input_dir, num_images, image_size=256):
        prompts = []
        samples = torch.empty(
            (0, 3, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        while len(prompts) < num_images:
            samples_i, prompts_i = self.try_get_raw_images(num_images - len(prompts))
            prompts.extend(prompts_i)
            samples = torch.cat((samples, samples_i)).view(
                -1, 3, self.image_size, self.image_size
            )
        return samples, prompts, image_size

    def post_process_raw(self, x):
        image, image_size = x
        return transforms.Resize((image_size, image_size), antialias=None)(image)

    def _encode_batch(self, x_batch, msg_batch):
        return self.gen_from_prompts(x_batch, add_watermark=True)

    def encode(self, x, with_grad=False):
        samples, orig_size = x
        encoded = torch.empty(
            (0, 3, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        pout = []
        n_batch = int(np.ceil(len(samples) / self.batch_size))
        for step in range(n_batch):
            imgs = samples[step * self.batch_size : (step + 1) * self.batch_size]
            msg_batch = (
                self.watermark.repeat(len(imgs), 1)
                if self.watermark is not None
                else None
            )
            if not with_grad:
                with torch.no_grad():
                    encoded_image_batch, pbatch = self._encode_batch(imgs, msg_batch)
            else:
                encoded_image_batch, pbatch = self._encode_batch(imgs, msg_batch)
            encoded = torch.cat((encoded, encoded_image_batch)).view(
                -1, 3, self.image_size, self.image_size
            )
            pout.extend(pbatch)
        return transforms.Resize((orig_size, orig_size), antialias=None)(encoded), pout

    def try_get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images, prompts, image_size = self.get_raw_images(
            input_dir, num_images, image_size=image_size
        )
        encoded, prompt_encoded = self.encode((prompts, image_size))
        raw_images = [
            r.unsqueeze(0) for r, p in zip(raw_images, prompts) if p in prompt_encoded
        ]
        raw_images = (
            torch.cat(raw_images).view(-1, 3, image_size, image_size)
            if len(raw_images) > 0
            else torch.empty(
                (0, 3, self.image_size, self.image_size),
                device=self.device,
                dtype=torch.float32,
            )
        )
        return self.post_process_raw((raw_images, image_size)), encoded

    def get_watermarked_images(self, input_dir, num_images, image_size=256):
        raw_images = torch.empty(
            (0, 3, image_size, image_size), device=self.device, dtype=torch.float32
        )
        encoded = torch.empty(
            (0, 3, image_size, image_size), device=self.device, dtype=torch.float32
        )
        while raw_images.shape[0] < num_images:
            raw_images_i, encoded_i = self.try_get_watermarked_images(
                input_dir, num_images - raw_images.shape[0], image_size
            )
            raw_images = torch.cat((raw_images, raw_images_i)).view(
                -1, 3, image_size, image_size
            )
            encoded = torch.cat((encoded, encoded_i)).view(
                -1, 3, image_size, image_size
            )
        return raw_images, encoded

    def _decode_batch_raw(self, x):
        trials = 0
        x = self.to_vimage(x)
        out = []
        result = "INVALID"

        for xi in x:
            while True:
                try:
                    result = self.decoder.verify_image(xi).watermark_verification_result
                    break
                except exceptions.ResourceExhausted as e:
                    trials += 1
                    if trials == self.api_trials:
                        logging.error(
                            "Google's RPC seems to be unresponsive. Aborting now. Please rerun the "
                            "attack with a different seed to continue with a fresh set of samples."
                        )
                        sys.exit(1)
                    time.sleep(self.trial_interval)
                    continue
                except Exception as e:
                    break

            if result not in ["ACCEPT", "REJECT"]:
                logging.error(
                    f"Google's RPC couldn't return a proper wm verificatio result: {result}. Aborting "
                    f"now. Please rerun the attack with a different seed to continue with a fresh set "
                    f"of samples."
                )
                sys.exit(1)

            out.append(1.0 if result == "ACCEPT" else 0.0)
        return torch.tensor(out, dtype=torch.float32, device=self.device).view(
            len(x), -1
        )

    def is_detected(self, accs):
        return accs > self.acceptance_thresh

    def _decode_batch(self, x_batch, msg_batch):
        return self._decode_batch_raw(x_batch)

    def err(self, x_batch, msg_batch):
        return torch.abs(x_batch - 1.0).view(-1)
