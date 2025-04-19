from typing import Tuple


class ModelConfig:
    def __init__(self):
        self.log_dir: str = './runs2'
        self.ckpt_path: str = './ckpts2'
        self.saved_ckpt_path: str = None
        self.world_size: int = 1
        self.lr: float = 0.0001
        self.num_epochs: int = 200
        self.log_interval: int = 400
        self.num_encoded_bits: int = 100
        self.image_shape: Tuple[int, int] = (256, 256)
        self.num_down_levels: int = 4
        self.num_initial_channels: int = 32
        self.batch_size: int = 16
        self.beta_min: float = 0.0001
        self.beta_max: float = 10.0
        self.beta_start_epoch: float = 1
        self.beta_epochs: int = 20
        self.warmup_epochs: int = 1
        self.discriminator_feature_dim: int = 16
        self.num_discriminator_layers: int = 4
        self.watermark_hidden_dim: int = 16
        self.psnr_threshold: float = 55.0
        self.enc_mode: str = "uuid" 
        self.ecc_t: int = 16
        self.ecc_m: int = 8
        self.num_classes: int = 2
        self.beta_transform: float = 0.5
        self.num_noises: int = 2
        self.noise_start_epoch: int = 15