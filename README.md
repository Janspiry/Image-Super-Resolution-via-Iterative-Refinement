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

### Result 

- ##### 16×16 -> 128×128 on FFHQ-CelebaHQ（Preview)

<img src="./misc/show.png" alt="show" style="zoom:100%;" />

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



### Reference

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
2. [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)



