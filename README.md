# UnMarker: A Universal Attack on Defensive Image Watermarking

Official PyTorch implementation of our IEEE S&P 2025 paper: ["UnMarker: A Universal Attack on Defensive Image Watermarking"](https://arxiv.org/abs/2405.08363).

Andre Kassis, Urs Hengartner

Contact: akassis@uwaterloo.ca


Abstract: *Reports regarding the misuse of Generative AI (GenAI) to create deepfakes are frequent. Defensive watermarking enables GenAI providers to hide fingerprints in their images and use them later for deepfake detection. Yet, its potential has not been fully explored. We present UnMarker--- the first practical universal attack on defensive watermarking. Unlike existing attacks, UnMarker requires no detector feedback, no unrealistic knowledge of the watermarking scheme or similar models, and no advanced denoising pipelines that may not be available. Instead, being the product of an in-depth analysis of the watermarking paradigm revealing that robust schemes must construct their watermarks in the spectral amplitudes, UnMarker employs two novel adversarial optimizations to disrupt the spectra of watermarked images, erasing the watermarks. Evaluations against SOTA schemes prove UnMarker's effectiveness. It not only defeats traditional schemes while retaining superior quality compared to existing attacks but also breaks semantic watermarks that alter an image's structure, reducing the best detection rate to 43\% and rendering them useless. To our knowledge, UnMarker is the first practical attack on semantic watermarks, which have been deemed the future of defensive watermarking. Our findings show that defensive watermarking is not a viable defense against deepfakes, and we urge the community to explore alternatives.*

### Acknowledgment

The code for the different watermarking schemes was adapted from the corresponding works or later works that reproduced the authors' results. Minor changes were made only to allow the integration of all systems into a unified framework. The pre-trained models are those published by the original authors. Specifically, we have the following schemes:
- <mark>StegaStamp</mark>: [Invisible Hyperlinks in Physical Photographs](https://arxiv.org/pdf/1904.05343) by Tancik et al. Taken from [StegaStamp](https://github.com/tancik/StegaStamp).
- <mark>StableSignature</mark>: [The stable signature: Rooting watermarks in latent diffusion models](https://arxiv.org/pdf/2303.15435). Taken from [stable_signature](https://github.com/facebookresearch/stable_signature).
- <mark>TreeRing</mark>: [Fingerprints for Diffusion Images that are Invisible and Robust](https://arxiv.org/pdf/2305.20030) by Wen et al. Taken from [tree-ring-watermark](https://github.com/YuxinWenRick/tree-ring-watermark).
- <mark>Yu1</mark>: [Responsible disclosure of generative models using scalable fingerprinting](https://arxiv.org/pdf/2012.08726) by Yu et al. Taken from [ScalableGANFingerprints](https://github.com/ningyu1991/ScalableGANFingerprints).
- <mark>Yu2</mark>: [Rooting deepfake attribution in training data](https://arxiv.org/pdf/2007.08457) by Yu et al. Taken from [ArtificialGANFingerprints](https://github.com/ningyu1991/ArtificialGANFingerprints).
- <mark>HiDDeN</mark>: [Hiding Data With Deep Networks](https://arxiv.org/pdf/1807.09937) by Zhu et al. Taken from [WEvade](https://github.com/zhengyuan-jiang/WEvade).
- <mark>PTW</mark>: The [Pivotal Tuning Watermarking](https://arxiv.org/pdf/2304.07361) scheme by Lukas & Kerschbaum.
Taken from [gan-watermark](https://github.com/nilslukas/gan-watermark).

The baseline regeneration attacks were constructed based on the description from [Invisible Image Watermarks Are Provably Removable Using Generative AI](https://arxiv.org/pdf/2306.01953) by Zhao et al. Specifically, the DiffusionAttack uses the diffusion-based purification backbone which was adapted from [DiffPure](https://github.com/NVlabs/DiffPure). We use the [GuidedModel](https://arxiv.org/pdf/2105.05233) by Dhariwal & Nichol for the attack. For the VAEAttack, we use the Bmshj2018 VAE from [CompressAI](https://github.com/InterDigitalInc/CompressAI).

### Citation

If you find our repo helpful, please consider citing it:

@INPROCEEDINGS {,
author = { Kassis, Andre and Hengartner, Urs },
booktitle = { 2025 IEEE Symposium on Security and Privacy (SP) },
title = {{ UnMarker: A Universal Attack on Defensive Image Watermarking }},
year = {2025},
doi = {10.1109/SP61157.2025.00005},
}

### News

**Newly supported watermark schemes (post-paper):** we added evaluations for the following recent schemes to further demonstrate UnMarker’s broad effectiveness:

- `Gs` — [Gaussian Shading: Provable Performance-Lossless Image Watermarking for Diffusion Models](https://arxiv.org/abs/2404.04956) (Yang et al.). Implementation: [Gaussian-Shading](https://github.com/bsmhmmlf/Gaussian-Shading).
- `Prc` — [An Undetectable Watermark for Generative Image Models](https://arxiv.org/pdf/2410.07369) (Gunn et al.). Implementation: [PRC-Watermark](https://github.com/XuandongZhao/PRC-Watermark).
- `Vine` — [Robust Watermarking Using Generative Priors Against Image Editing: From Benchmarking to Advances](https://arxiv.org/abs/2410.18775) (Lu et al.). Implementation: [VINE](https://github.com/Shilin-LU/VINE).

Special thanks to [@manaswineegupta](https://github.com/manaswineegupta) for integrating these schemes.

**Google SynthID:** UnMarker reduces detection on Google’s image watermarking scheme [SynthID](https://deepmind.google/science/synthid/), achieving a **79% attack success rate** in our experiments (i.e., detection drops from ~100% to ~21%).

- We disclosed these results to Google; they acknowledged the findings and issued a bounty via the [VRP](https://bughunters.google.com/) program.
- Our experiments used the Vertex AI Python SDK to (a) generate watermarked images with Imagen and (b) evaluate detections. We include evaluation code and instructions (see `SynthID (Vertex AI) — setup & usage`).
- **Note:** Google appears to have restricted access to the verification model via the SDK since our runs. Automated reproduction with our harness may no longer work. Manual verification via the SynthID web UI remains possible, and UnMarker can be evaluated this way. However, minor code changes are needed to save raw attack outputs for manual uploads. If interested, please open an issue.

### Requirements
- A high-end NVIDIA GPU with >=32 GB memory.
- [CUDA=12](https://docs.nvidia.com/datacenter/tesla/driver-installation-guide/index.html) driver must be installed.
- [Anaconda](https://docs.anaconda.com/anaconda/install/) must be installed.
- ~30GB of storage space to save the pretrained models and datasets.

### Setup

#### A) Installation 

```console
conda create -n unmarker python=3.10
conda activate unmarker
git clone https://github.com/andrekassis/ai-watermark.git
cd ai-watermark
./install.sh
```

#### B) Download the pretrained models and datasets and extract them here 
This is handled automatically by simply running ```./download_data_and_models.sh```.

### Usage

#### A) Running Attacks

The file attack.py is responsible for running the various attacks.

Run: ```python attack.py -o OUTPUT_DIR -a ATTACK_NAME -e SCHEME_NAME```

The different attack names (including other baseline attacks) and scheme names can be found in attack.py-- Please refer to this file for all available attack and scheme names. OUTPUT_DIR should be the name of the output directory where you wish to save the results (and attack images). By default, the attack runs on 100 images that are first watermarked and then targeted (to remove the watermark). You may change this number by providing the option ```--total_imgs TOTAL``` to attack.py as well, with TOTAL being the alternative number of images you want to use. That said, the provided sample datasets for **HiDDeN** and **StegaStamp** (i.e., the subsets of [COCO](https://arxiv.org/pdf/1405.0312) and [CelebA-HQ](https://arxiv.org/pdf/1710.10196)) you download and save in "datasets" only contain 100 images and therefore a larger number will not be possible. You may download additional records from these datasets manually, however, and place them in the appropriate locations to experiment with more samples. You may also provide your own data instead by simply changing the path to the directory containing the input images to watermark and attack. This option in in the configuration files <mark>attack_configs/StegaStamp.yaml</mark> and <mark>attack_configs/HiDDeN.yaml</mark> under <mark>input_dir</mark>.

*Note that the baseline attacks "Noise" (which corresponds to random noise addition to the image) and SuperResolution (that performs downscaling and then restores the original image size using a super-resolution diffusion model) were not considered in the paper for time limitations and for not being any more effective than the remaining baselines. However, you may still run evaluations with these baselines as well if you wish.*

#### B) Attack Parameters

The attack parameters are in the directory attack_configs. For each scheme, the directory contains a ".yaml" file with the scheme's name that contains the parameters of the different attacks. You should not change the parameters for UnMarker or the DiffusionAttack (unless you explicitly intend to do so). However, you may still change the parameter <mark>**loss_thresh**</mark> for UnMarker or <mark>**t**</mark> for the DiffusionAttack to control the output image quality as you need. Note that for UnMarker, stage1_args correspond to the parameters of the first optimization stage (i.e., high-frequency), while stage2_args are the parameters for the low-frequency optimizations. For the remaining attacks, the files include specific default parameters with which they are instantiated. Other parameters that were considered in the paper are commented out with \# and can be used instead of the defaults as well.

##### Stage Selection

You can also choose which of UnMarker's stages to use for the attack. As explained in the paper, the low-frequency stage is effective against semantic watermarks, while the high-frequency stage is suitable for non-semantic schemes. Combining both stages yields samples of acceptable quality for high-resolution images (i.e., StableSignature, StegaStamp, and TreeRing), but it can still leave visible traces. While these traces, if at all visible, generally appear as slightly hazy backgrounds that do not interfere with the main content and, therefore, retain similarity/quality, they may be undesirable. One can potentially reduce their effects by selecting only the suitable stage for the relevant watermarking scheme: low-frequency for StegaStamp and TreeRing and high-frequency for StableSignature (although high-frequency modifications can boost the performance further against TreeRing as well). As such, the attack is run with these stages only by default for these schemes. If you wish to enable all stages, simply change the value of the <mark>**stage_selector**</mark> entry under UnMarker's parameters in the relevant scheme's attack configuration file as follows:
```
stage_selector: [preprocess, stage1, stage2]
```
Here, ```preprocess``` refers to cropping. We note that other changes, such as modifying the learning rates, visual loss thresholds, or even the visual loss function itself, may even yield better results and enhanced image quality. Feel free to experiment with these configurations.

#### C) Parsing results

##### Output Files

The output attack images will be saved to <mark>OUT_DIR/images</mark>. For general-purpose schemes that accept input images, you will find a triplet of images for each input corresponding to the original, watermarked, and attacked (removed) images. For the other schemes, you will find pairs of watermarked and removed images only. Each output in this directory is named <mark>img_IDX.png</mark>, where IDX is the index (position) of the corresponding input in the evaluation.

In <mark>OUT_DIR/log.txt</mark>, you will find per-input statistics. For each input with position <mark>IDX</mark> in the evaluation, you will find a record of the following format:
```console
img_IDX - [orig: WATERMARK_BIT_ACCURACY_IN_THE_NON_WATERMARKED_IMAGE], watermarked: WATERMARK_BIT_ACCURACY_IN_THE_WATERMARKED_IMAGE, removed: WATERMARK_BIT_ACCURACY_IN_THE_ATTACKED_IMAGE, similarity score: LPIPS_SIMILARITY_SCORE
```

where LPIPS_SIMILARITY_SCORE denotes the <mark>**lpips**</mark> similarity between the watermarked and attacked image. Note that the entry <mark>[orig: WATERMARK_BIT_ACCURACY_IN_THE_NON_WATERMARKED_IMAGE]</mark> will only be present for general-purpose schemes. Lower <mark>**lpips**</mark> scores indicated better attack quality, while lower bit accuracy for the attacked images means the attack is successful in removing the watermark. For watermarked images, the bit accuracies should be high.

In <mark>OUT_DIR/aggregated_results.yaml</mark>, you will find the aggregated statistics for your experiment (these will also be printed to the screen). This <mark>**yaml**</mark> file contains a dictionary with the following entries:
- <mark>**attack**</mark>: The attack's name.
- <mark>**scheme**</mark>: The scheme's name.
- <mark>**detection threshold**</mark>: The threshold used to determine whether the watermark has been detected. Please refer to the paper for details on how the thresholds were determined.
- <mark>**lpips**</mark>: The average lpips similarity scores for all watermarked-attacked input pairs.
- <mark>**FID**</mark>: The FID distance between the watermarked and attacked samples.
- <mark>**detection rates**</mark>: The scheme's average detection rates for all images. This is a dictionary with the following entries:
    - <mark>*orig*</mark>: Average watermark detection rates in all original (non-watermarked) images. This entry is only present for general-purpose schemes. The lower this number, the better the scheme's ability to reject false positives.
    - <mark>*watermarked*</mark>: Average watermark detection rates in all watermarked images. The higher this number is, the better the scheme's ability to detect the watermark (without any attacks).
    - <mark>*removed*</mark>:  Average watermark detection rates in all attacked images. The lower this number is, the better the attack's performance.

##### Screen Logs

While the attack is running, the cumulative detection rates are constantly logged to the screen as well (under "orig," "watermarked," and "removed"). Note that the detection rates are derived from the individual bit accuracies above based on the detection thresholds, as explained in the paper.

For UnMarker, optimization includes the low-frequency stage and the high-frequency stage. As these stages are iterative, the attack also logs the statistics for the sample under optimization after each iteration to the screen. The message printed is as follows:
```
UnMarker-STAGE - Step STEP, best loss: BEST_LOSS, curr loss: CURR_LOSS, dist: LPIPS, reg_loss: L2_REGULARIZATION_LOSS, [filter_loss: FILTER_LOSS] detection acc: BIT_ACC, attack_success: NOT_DETECTED
```
where the different entries have the following meanings:
- STAGE: Name of the optimization stage of UnMarker-- either "low_freq" or "high_freq".
- STEP: The current binary search step. Refer to the paper for details.
- BEST_LOSS: The best (maximum) spectral loss attained by the attack stage thus far (within the constraints).
- CURR_LOSS: The spectral loss at the current iteration.
- LPIPS: The <mark>**lpips**</mark> distance of the optimized sample at the current iteration from the input to the stage.
- L2_REGULARIZATION_LOSS: The <mark>**l2**</mark> regularization loss-- Refer to the paper for details.
- FILTER_LOSS: UnMarker's filter loss. This entry is only present for the "low_freq" stage.
- BIT_ACC: The bit accuracy of the extracted watermark from the adversarial sample at the current step-- Lower values indicate the attack is more successful.
- NOT_DETECTED: Assigned 0 or 1 based on whether the watermark is no longer detected (1). This means that the current BIT_ACC is below the required threshold for detection.

Note that evaluating the watermark can be costly at each optimization iteration, which would unnecessarily slow down the optimization despite it being unnecessary (as this information is not required for the attack but is merely for logging. While this is not an issue for most systems, TreeRing's watermark evaluation is extremely slow compared to all other schemes. As such, you may change the parameter <mark>eval_interval</mark> under <mark>progress_bar_args</mark> in the attack configuration file for UnMarker, choosing a large interval instead and thereby instructing UnMarker to log these watermark statistics less frequently (or never).

### SynthID (Vertex AI) — setup & usage (optional)

To generate watermarked images with Google's Imagen via VertexAI and verify the watermarks with SynthID, some dependencies and tools must be installed.

**Prerequisites**
1. Create a Google Cloud project **with billing enabled**.  
   ↳ https://cloud.google.com/billing/docs/how-to/verify-billing-enabled#confirm_billing_is_enabled_on_a_project
2. Enable the **Vertex AI** API for that project.  
   ↳ https://console.cloud.google.com/flows/enableapi?apiid=aiplatform.googleapis.com

**Install**
```bash
# Runs dependency checks and gcloud setup
./install_synthid.sh
```

When prompted “Do you want to configure a default Compute Region and Zone? (Y/n)”, choose Y and select a zone from the menu.

Run: ```python attack.py -a UnMarker -e SynthID -o OUTPUT_DIR``` (replace OUTPUT_DIR with your output directory of choice).

*Important limitation:* Since our original experiments, Google appears to have restricted access to the SynthID verification model via the Vertex AI Python SDK. As a result, the automated end-to-end tests in this repo will not run. You can still evaluate UnMarker manually by uploading outputs to the SynthID web UI. If you need the codepath to emit raw attack images for manual upload, please open an issue.
