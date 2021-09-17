# Image Super-Resolution via Iterative Refinement

[Paper](https://arxiv.org/pdf/2104.07636.pdf ) |  [Project](https://iterative-refinement.github.io/ )



## Brief

This is a unoffical implementation about **Image Super-Resolution via Iterative Refinement(SR3)** by **Pytorch**.

There are some implement details with paper description, which maybe different with actual `SR3` structure due to details missing.

- We used the ResNet block and channel concatenation style like vanilla `DDPM`.
- We used the attention mechanism in low resolution feature(16×16) like vanilla `DDPM`.
- We encoding the $\gamma$ as `FilM` strcutrue did in `WaveGrad`, and embedding it without affine transformation.



## Status

### Conditional generation(super resolution)

- [x] 16×16 -> 128×128 on FFHQ-CelebaHQ
- [x] 64×64 -> 512×512 on FFHQ-CelebaHQ

### Unconditional generation

- [x] 128×128 face generation on FFHQ
- [ ] ~~1024×1024 face generation by a cascade of 3 models~~

### Training Step

- [x] log / logger
- [x] metrics evaluation
- [x] multi-gpu support
- [x] resume training / pretrained model
- [x] validate alone script



## Results

*Note:*  We set the maximum reverse steps budget to 2000 now. Limited to model parameters in `Nvidia 1080Ti`, **image noise** and **hue deviation** occasionally appears in high-resolution images, resulting in low scores.  There are a lot room to optimization.

| Tasks/Metrics        | SSIM(+) | PSNR(+) | FID(-)  | IS(+)   |
| -------------------- | ----------- | -------- | ---- | ---- |
| 16×16 -> 128×128 | 0.675       | 23.26    | - | - |
| 64×64 -> 512×512     | 0.445 | 19.87 | - | - |
| 128×128 | - | - | | |
| 1024×1024 | - | - |      |      |

- #### 16×16 -> 128×128 on FFHQ-CelebaHQ [[More Results](https://drive.google.com/drive/folders/1Vk1lpHzbDf03nME5fV9a-lWzSh3kMK14?usp=sharing)]

| <img src="./misc/sr_process_16_128_0.png" alt="show" style="zoom:90%;" /> |  <img src="./misc/sr_process_16_128_1.png" alt="show" style="zoom:90%;" />    |   <img src="./misc/sr_process_16_128_2.png" alt="show" style="zoom:90%;" />   |
| ------------------------------------------------------------ | ---- | ---- |

- #### 64×64 -> 512×512 on FFHQ-CelebaHQ [[More Results](https://drive.google.com/drive/folders/1yp_4xChPSZUeVIgxbZM-e3ZSsSgnaR9Z?usp=sharing)]

| <img src="./misc/sr_64_512_0_inf.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_0_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_0_hr.png" alt="show" style="zoom:90%;" /> |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="./misc/sr_64_512_1_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_2_sr.png" alt="show" style="zoom:90%;" /> | <img src="./misc/sr_64_512_3_sr.png" alt="show" style="zoom:90%;" /> |

- #### 128×128 face generation on FFHQ [[More Results](https://drive.google.com/drive/folders/13AsjRwDw4wMmL0bK7wPd2rP7ds7eyAMh?usp=sharing)]

| <img src="./misc/sample_process_128_0.png" alt="show" style="zoom:90%;" /> |  <img src="./misc/sample_process_128_1.png" alt="show" style="zoom:90%;" />    |   <img src="./misc/sample_process_128_2.png" alt="show" style="zoom:90%;" />   |
| ------------------------------------------------------------ | ---- | ---- |



## Usage

### Pretrained Model

This paper is based on "Denoising Diffusion Probabilistic Models", and we build both `DDPM/SR3` network structure, which use timesteps/gama as model embedding input, respectively. In our experiments, `SR3` model can achieve better visual results with same reverse steps and learning rate. You can select the json files with annotated suffix names to train different model.

| Tasks                             | Google Drive                                                 |
| --------------------------------- | ------------------------------------------------------------ |
| 16×16 -> 128×128 on FFHQ-CelebaHQ | [SR3](https://drive.google.com/drive/folders/12jh0K8XoM1FqpeByXvugHHAF3oAZ8KRu?usp=sharing) |
| 64×64 -> 512×512 on FFHQ-CelebaHQ | [SR3](https://drive.google.com/drive/folders/1mCiWhFqHyjt5zE4IdA41fjFwCYdqDzSF?usp=sharing) |
| 128×128 face generation on FFHQ   | [SR3](https://drive.google.com/drive/folders/1ldukMgLKAxE7qiKdFJlu-qubGlnW-982?usp=sharing) |

```python
# Download the pretrain model and edit [sr|sample]_[ddpm|sr3]_[resolution option].json about "resume_state":
"resume_state": [your pretrain model path]
```

### Data Prepare

#### New Start

If you didn't have the data, you can prepare it by following steps:

- [FFHQ 128×128](https://github.com/NVlabs/ffhq-dataset) | [FFHQ 512×512](https://www.kaggle.com/arnaud58/flickrfaceshq-dataset-ffhq)
- [CelebaHQ 256×256](https://www.kaggle.com/badasstechie/celebahq-resized-256x256) | [CelebaMask-HQ 1024×1024](https://drive.google.com/file/d/1badu11NqxGf6qM3PTTooQDJvQbejgbTv/view)

Download the dataset and prepare it in **LMDB** or **PNG** format using script.

```python
# Resize to get 16×16 LR_IMGS and 128×128 HR_IMGS, then prepare 128×128 Fake SR_IMGS by bicubic interpolation
python prepare.py  --path [dataset root]  --out [output root] --size 16,128 -l
```

then you need to change the datasets config to your data path and image resolution: 

```json
"datasets": {
    "train": {
        "dataroot": "dataset/ffhq_16_128", // [output root] in prepare.py script
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "lmdb", //lmdb or img, path of img files
    },
    "val": {
        "dataroot": "dataset/celebahq_16_128", // [output root] in prepare.py script
    }
},
```

#### Own Data

You also can use your image data by following steps.

At first, you should organize images layout like this:

```shell
# set the high/low resolution images, bicubic interpolation images path
dataset/celebahq_16_128/
├── hr_128
├── lr_16
└── sr_16_128
```

then you need to change the dataset config to your data path and image resolution: 
```json
"datasets": {
    "train|val": {
        "dataroot": "dataset/celebahq_16_128",
        "l_resolution": 16, // low resolution need to super_resolution
        "r_resolution": 128, // high resolution
        "datatype": "img", //lmdb or img, path of img files
    }
},
```

### Training/Resume Training

```python
# Use sr.py and sample.py to train the super resolution task and unconditional generation task, respectively.
# Edit json files to adjust network structure and hyperparameters
python sr.py -p train -c config/sr_sr3.json
```

### Test/Evaluation

```python
# Edit json to add pretrain model path and run the evaluation 
python sr.py -p val -c config/sr_sr3.json

# Quantitative evaluation alone using SSIM/PSNR metrics on given result root
python eval.py -p [result root]
```

### Inference Alone

Set the HR (vanilla high resolution images), SR (images need processed) image path like step in `Own Data`. HR directory contexts can be copy from SR, and LR directory  is unnecessary. 

```python
# run the script
python infer.py -c [config file]
```



## Acknowledge

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
- https://github.com/hejingwenhejingwen/AdaFM



