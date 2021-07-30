## Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) 	

[Project](https://iterative-refinement.github.io/ )



### Brief

This is a unoffical implementation about **Image Super-Resolution via Iterative Refinement** by **Pytorch**.

Code will come soon.



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
- [ ] resume training
- [ ] multi-gpu support



### Usage

#### Data Prepare

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256)

```python
# Resize to get 16×16 MINI_IMGS and 128×128 HR_IMGS, then prepare 128×128 LR_IMGS by bicubic interpolation
python prepare.py  --path ffhq_128  --out ffhq --size 16,128
```

#### Train

```

```



#### Test

```

```





### Reference

1. [Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
2. [Image Super-Resolution via Iterative Refinement](https://arxiv.org/pdf/2104.07636.pdf)



