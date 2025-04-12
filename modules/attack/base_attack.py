import os
import shutil
import torch
import numpy as np
from tqdm import trange
from PIL import Image, ImageFont, ImageDraw, ImageOps
from torchvision import transforms
import matplotlib.pyplot as plt
from watermarkers import init_watermarker

from torcheval.metrics import FrechetInceptionDistance
import lpips

from ..logs import get_logger

logger = get_logger()


class LpipsVGG(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        loss_fn = (
            lpips.LPIPS(
                net="vgg", model_path="pretrained_models/vgg.pth", verbose=False
            )
            .to(device)
            .eval()
        )
        self.loss_fn = lambda x, y: loss_fn(x, y, normalize=True).view(-1, 1)

    def forward(self, x, y):
        return self.loss_fn(x, y)


class BaseAttack:
    def __init__(
        self,
        evaluator,
        input_dir,
        output_dir,
        batch_size=1,
        device="cuda",
    ):
        self.e = evaluator
        self.evaluator = init_watermarker(
            evaluator, batch_size=batch_size, device=device
        )
        self.setup_dirs(evaluator, input_dir, output_dir)
        self.image_size = self.evaluator.image_size
        self.batch_size = batch_size
        self.device = device

        self.sim_loss = LpipsVGG(self.device)
        self.FID = FrechetInceptionDistance().to(device)
        self.total_sim = []

    def calc_sim_loss(self, removed, watermarked):
        self.FID.update(torch.clamp(watermarked, 0.0, 1.0), True)
        self.FID.update(torch.clamp(removed, 0.0, 1.0), False)

        return self.sim_loss(removed, watermarked).detach().item()

    def setup_dirs(self, evaluator, input_dir, output_dir):
        self.inputs_dir = input_dir
        self.total_available_images = (
            len(os.listdir(self.inputs_dir)) if self.inputs_dir is not None else -1
        )

        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir if output_dir is not None else None

        if self.output_dir is not None:
            if os.path.exists(self.output_dir):
                shutil.rmtree(self.output_dir)
            os.makedirs(self.output_dir)
            os.makedirs(os.path.join(self.output_dir, "images"))

    def evaluate(self, x):
        return self.evaluator.evaluate(x)[0] if x is not None else None

    def __extend(self, lst, element):
        if element is not None:
            lst.extend(list(element))

    def __to_tensor(self, lst, shape):
        if len(lst) == 0:
            return None
        return torch.concat(lst).view(*shape)

    def __call__(self, num_images):
        if self.total_available_images != -1:
            if num_images > self.total_available_images:
                logger.error(
                    f"Requested attack on {num_images} images. However, only {self.total_available_images} "
                    f"are available in the dataset. Aboring!"
                )
                exit(1)

        n_batch = int(np.ceil(num_images / self.batch_size))
        out_lst = [[], [], [], []]
        acc_lst = [[], [], [], []]
        sim_lst = []
        self.FID.reset()
        pbar = trange(n_batch)
        pbar.set_description("Processed samples")

        for step in pbar:
            batch = np.min([num_images - step * self.batch_size, self.batch_size])
            orig, watermarked = self.evaluator.get_watermarked_images(
                self.inputs_dir, batch, image_size=self.image_size
            )

            orig, watermarked = (
                orig.detach() if orig is not None else orig,
                watermarked.detach(),
            )
            denoised, removed = self.do_batch(watermarked.clone().detach())
            removed = self.post_process_removed(removed)

            if watermarked is not None and removed is not None:
                sim_lst.append(self.calc_sim_loss(removed, watermarked))
            for o, o_lst in zip([orig, watermarked, denoised, removed], out_lst):
                self.__extend(o_lst, o)

            orig = self.evaluator.get_unmarked_images(
                orig, self.inputs_dir, batch, self.image_size
            )

            for o, a_lst in zip([orig, watermarked, denoised, removed], acc_lst):
                self.__extend(a_lst, self.evaluate(o))

            out_shape = (-1, 3, out_lst[1][0].shape[-1], out_lst[1][0].shape[-1])
            orig, watermarked, denoised, removed = [
                self.__to_tensor(out, out_shape) for out in out_lst
            ]
            self.save_img_quartets(
                orig, watermarked, denoised, removed, start=self.batch_size * step
            )
            out_lst = [[], [], [], []]

            o_accs, w_accs, d_accs, r_accs = [
                np.array(acc) if len(acc) > 0 else None for acc in acc_lst
            ]
            all_names = self.get_output_names(o_accs, w_accs, d_accs, r_accs)
            accs = [acc for acc in [o_accs, w_accs, d_accs, r_accs] if acc is not None]
            acc_dict = {name: acc for name, acc in zip(all_names, accs)}

            self.write_logs(acc_dict, sim_lst)

            out = {
                name: float(np.mean(self.evaluator.is_detected(acc)))
                for name, acc in acc_dict.items()
            }
            #pbar.update(1)
            pbar.set_description(
                f"{', '.join([ f'{k}: {v}' for k, v in out.items() ])}. Processed samples"
            )
        sim = np.mean(np.array(sim_lst))
        fid = self.FID.compute().detach().item()
        return {"lpips": sim, "FID": fid, "detection rates": out}

    def write_logs(self, acc_dict, sim_lst):
        if self.output_dir is not None:
            out_f = os.path.join(self.output_dir, "log.txt")
            num_images = len(acc_dict[list(acc_dict.keys())[0]])
            stats = []
            for i in range(num_images):
                score = ", similarity score: " + str(sim_lst[i]) if sim_lst else ""
                stats.append(
                    f"img_{i} - "
                    + ", ".join([k + ": " + str(v[i]) for k, v in acc_dict.items()])
                    + score
                    + "\n"
                )
            with open(out_f, "w") as f:
                f.writelines(stats)

    def get_output_names(
        self, orig_imgs, watermarked_imgs, denoised_imgs, removed_imgs
    ):
        all_names = ["orig", "watermarked", "denoised", "removed"]
        return [
            all_names[i]
            for i, imgs in enumerate(
                (orig_imgs, watermarked_imgs, denoised_imgs, removed_imgs)
            )
            if imgs is not None
        ]

    def post_process_removed(self, imgs):
        return imgs

    def _prepare_for_save(self, imgs):
        return (
            [
                transforms.ToPILImage()(img.squeeze())
                for img in self.evaluator.resizer(imgs).detach().cpu()
            ]
            if imgs is not None
            else imgs
        )

    def save_img_quartets(
        self,
        orig_imgs,
        watermarked_imgs,
        denoised_imgs,
        removed_imgs,
        start=0,
    ):
        if self.output_dir is None:
            return
        output_dir = os.path.join(self.output_dir, "images")

        orig_imgs, watermarked_imgs, denoised_imgs, removed_imgs = (
            self._prepare_for_save(orig_imgs),
            self._prepare_for_save(watermarked_imgs),
            self._prepare_for_save(denoised_imgs),
            self._prepare_for_save(removed_imgs),
        )
        denoised_imgs = None
        all_imgs = [
            imgs
            for imgs in (orig_imgs, watermarked_imgs, denoised_imgs, removed_imgs)
            if imgs is not None
        ]
        all_names = self.get_output_names(
            orig_imgs, watermarked_imgs, denoised_imgs, removed_imgs
        )
        total_size = self.evaluator.image_size * len(all_names)
        for i, images in enumerate(zip(*all_imgs)):
            new_im = Image.new("RGB", (total_size, self.evaluator.image_size))
            x_offset = 0
            draw_offsets = [0]
            for im, n in zip(images, all_names):
                new_im.paste(im, (x_offset, 0))
                x_offset += self.evaluator.image_size
                draw_offsets.append(x_offset)
            draw_offsets = draw_offsets[:-1]
            new_im = ImageOps.expand(new_im, border=20, fill=(255, 255, 255))
            draw = ImageDraw.Draw(new_im)
            font = font = ImageFont.truetype("assets/arial.ttf", 15)
            for name, offset in zip(all_names, draw_offsets):
                draw.text(
                    (offset + int(self.evaluator.image_size / 2), 0),
                    name,
                    (0, 0, 0),
                    font=font,
                )
            new_im.save(os.path.join(output_dir, "img_" + str(i + start) + ".png"))
