## Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf )  [Project](https://iterative-refinement.github.io/ )



### Brief

This is a unoffical implementation about **Image Super-Resolution via Iterative Refinement** by **Pytorch**.

Code will come soon and pretrain model will be released after training.



### Todo

#### Conditional generation(super resolution)

- [ ] 16×16 -> 128×128 on FFHQ-CelebaHQ
- [ ] 64×64 -> 512×512 on FFHQ-CelebaHQ
- [ ] 64×64 -> 256×256 on ImageNet 

#### Unconditional generation

- [ ] 1024×1024 face generation by a cascade of 3 models

#### Training Step

- [x] log/logger
- [x] metrics evaluation
- [x] resume training 
- [x] multi-gpu support



### Result (preview version only)

We set the maximum reverse steps budget to 2000 now.

| Tasks/Metrics        | **SSIM** | **PSNR** |
| -------------------- | -------- | -------- |
| **16×16 -> 128×128** | 0.675    | 23.26    |
| 64×64 -> 512×512     |          |          |
| 1024×1024            |          |          |

- ##### 16×16 -> 128×128 on FFHQ-CelebaHQ 

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

#### Train

```python
# Edit basic_sr.json to adjust network function and hyperparameters
python run.py -p train -c config/basic_sr.json
```

#### Test

```python
# Edit basic_sr.json to add pretrain model path 
python run.py -p val -c config/basic_sr.json
```

#### Evaluation
```python
# Quantitative evaluation using SSIM/PSNR metrics on given dataset root
python eval.py -p [dataset root]
```



### Reference

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
2. [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)



