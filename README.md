## Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf )  [Project](https://iterative-refinement.github.io/ )



### Brief

This is a unoffical implementation about **Image Super-Resolution via Iterative Refinement(SR3)** by **Pytorch**.

There are some implement details with paper description, which maybe different with actual `SR3` structure due to details missing.

- We used the Res-Net block and channel concatenation style in vanilla `DDPM`.
- We used the attention mechanism in low resolution feature(16×16) like vanilla `DDPM`.
- We encoding the gama as `FilM` strcutrue did in `Wave Grad`, and embedding it without affine transformation.



### Status

#### Conditional generation(super resolution)

- [x] 16×16 -> 128×128 on FFHQ-CelebaHQ
- [ ] 64×64 -> 512×512 on FFHQ-CelebaHQ

#### Unconditional generation

- [ ] 128×128 face generation on FFHQ
- [ ] 1024×1024 face generation by a cascade of 3 models

#### Training Step

- [x] log/logger
- [x] metrics evaluation
- [x] multi-gpu support
- [x] resume training/pretrain model



### Results

We set the maximum reverse steps budget to 2000 now.

| Tasks/Metrics        | SSIM(+) | PSNR(+) | FID(-)  | IS(+)   |
| -------------------- | ----------- | -------- | ---- | ---- |
| 16×16 -> 128×128 | 0.675       | 23.26    |      |      |
| 64×64 -> 512×512     |             |          |      |      |
| 1024×1024            |             |          |      |      |

- ##### 16×16 -> 128×128 on FFHQ-CelebaHQ [[More Results](https://drive.google.com/drive/folders/1Vk1lpHzbDf03nME5fV9a-lWzSh3kMK14?usp=sharing)]

| <img src="./misc/sr_process_16_128_0.png" alt="show" style="zoom:90%;" /> |  <img src="./misc/sr_process_16_128_1.png" alt="show" style="zoom:90%;" />    |   <img src="./misc/sr_process_16_128_2.png" alt="show" style="zoom:90%;" />   |
| ------------------------------------------------------------ | ---- | ---- |

### Usage

#### Data Prepare

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python prepare.py  --path [dataset root]  --out [output root] --size 16,128 -l
```



#### Pretrain Model

This paper is based on "Denoising Diffusion Probabilistic Models", and we build both `DDPM/SR3` network structure due to the lack of network details, which use timesteps/gama as model embedding input, respectively.  You can use  json files with their own suffix names to train different models, and they have similar performance according to our experiments.

| Tasks                             | Google Drive                                                 | Aliyun Drive                                     |
| --------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [SR3](https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu?usp=sharing) | [SR3](https://www.aliyundrive.com/s/EJXxgxqKy9z) |

```
# Download the pretrain model and edit basic_ddpm.json about "resume_state":
"resume_state": [your pretrain model path]
```

We have not trained the model until converged for time reason, which means there are a lot room to optimization.



#### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit sr_sr3.json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

#### Test/Evaluation

```python
# Edit sr_sr3.json to add pretrain model path 
python sr.py -p val -c config/sr_sr3.json
```

#### Evaluation Alone
```python
# Quantitative evaluation using SSIM/PSNR metrics on given dataset root
python eval.py -p [dataset root]
```



### Acknowledge

Our work is based on the following theoretical works:

- [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
- [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)
- [WaveGrad: Estimating Gradients for Waveform Generation](https://arxiv.org/abs/2009.00713)
- [Large Scale GAN Training for High Fidelity Natural Image Synthesis](https://arxiv.org/abs/1809.11096)

and we are benefit a lot from following projects:

- https://github.com/bhushan23/BIG-GAN
- https://github.com/lmnt-com/wavegrad
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/lucidrains/denoising-diffusion-pytorch



