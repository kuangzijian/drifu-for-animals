# DRIFu: Differentiable Rendering and Implicit Function-based Single-View 3D Reconstruction****
This repository contains a pytorch implementation of "[NIFu: Neural Rendering and Implicit Function-based Single-View 3D Reconstruction](https://arxiv.org/abs/2311.13199)".

If you find the code useful in your research, please consider citing the paper.

```
@misc{kuang2023drifu,
      title={DRIFu: Differentiable Rendering and Implicit Function-based Single-View 3D Reconstruction}, 
      author={Zijian Kuang and Lihang Ying and Shi Jin},
      year={2023},
      eprint={2311.13199},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


This codebase provides: 
- test code
- training code
- data generation code

## Requirements
- Python 3
- [PyTorch](https://pytorch.org/) tested on 1.4.0
- json
- PIL
- skimage
- tqdm
- numpy
- cv2

for training and data generation
- [trimesh](https://trimsh.org/) with [pyembree](https://github.com/scopatz/pyembree)
- [pyexr](https://github.com/tvogels/pyexr)
- PyOpenGL
- freeglut (use `sudo apt-get install freeglut3-dev` for ubuntu users)
- (optional) egl related packages for rendering with headless machines. (use `apt install libgl1-mesa-dri libegl1-mesa libgbm1` for ubuntu users)

Warning: I found that outdated NVIDIA drivers may cause errors with EGL. If you want to try out the EGL version, please update your NVIDIA driver to the latest!!

## Windows demo installation instuction

- Install [miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Add `conda` to PATH
- Install [git bash](https://git-scm.com/downloads)
- Launch `Git\bin\bash.exe`
- `eval "$(conda shell.bash hook)"` then `conda activate my_env` because of [this](https://github.com/conda/conda-build/issues/3371)
- Automatic `env create -f environment.yml` (look [this](https://github.com/conda/conda/issues/3417))
- OR manually setup [environment](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533)
    - `conda create —name drifu python` where `drifu` is name of your environment
    - `conda activate`
    - `conda install pytorch torchvision cudatoolkit=10.1 -c pytorch`
    - `conda install pillow`
    - `conda install scikit-image`
    - `conda install tqdm`
    - `conda install -c menpo opencv`
- Download [wget.exe](https://eternallybored.org/misc/wget/)
- Place it into `Git\mingw64\bin`
- `sh ./scripts/download_trained_model.sh`
- Remove background from your image ([this](https://www.remove.bg/), for example)
- Create black-white mask .png
- Replace original from sample_images/
- Try it out - `sh ./scripts/test.sh`
- Download [Meshlab](http://www.meshlab.net/) because of [this](https://github.com/shunsukesaito/drifu/issues/1)
- Open .obj file in Meshlab


## Demo
Warning: The released model is trained with mostly upright standing scans with weak perspectie projection and the pitch angle of 0 degree. Reconstruction quality may degrade for images highly deviated from trainining data.
1. run the following script to download the pretrained models from the following link and copy them under `./drifu/checkpoints/`.
```
sh ./scripts/download_trained_model.sh
```

2. run the following script. the script creates a textured `.obj` file under `./drifu/eval_results/`. You may need to use `./apps/crop_img.py` to roughly align an input image and the corresponding mask to the training data for better performance. For background removal, you can use any off-the-shelf tools such as [removebg](https://www.remove.bg/).
```
sh ./scripts/test.sh
```

## Demo on Google Colab
If you do not have a setup to run drifu, we offer Google Colab version to give it a try, allowing you to run drifu in the cloud, free of charge. Try our Colab demo using the following notebook: 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1GFSsqP2BWz4gtq0e-nki00ZHSirXwFyY)

## Data Generation (Linux Only)
While we are unable to release the full training data due to the restriction of commertial scans, we provide rendering code using free models in [RenderPeople](https://renderpeople.com/free-3d-people/).
This tutorial uses `rp_dennis_posed_004` model. Please download the model from [this link](https://renderpeople.com/sample/free/rp_dennis_posed_004_OBJ.zip) and unzip the content under a folder named `rp_dennis_posed_004_OBJ`. The same process can be applied to other RenderPeople data.

Warning: the following code becomes extremely slow without [pyembree](https://github.com/scopatz/pyembree). Please make sure you install pyembree.

1. run the following script to compute spherical harmonics coefficients for [precomputed radiance transfer (PRT)](https://sites.fas.harvard.edu/~cs278/papers/prt.pdf). In a nutshell, PRT is used to account for accurate light transport including ambient occlusion without compromising online rendering time, which significantly improves the photorealism compared with [a common sperical harmonics rendering using surface normals](https://cseweb.ucsd.edu/~ravir/papers/envmap/envmap.pdf). This process has to be done once for each obj file.
```
python -m apps.prt_util -i {path_to_rp_dennis_posed_004_OBJ}
```

2. run the following script. Under the specified data path, the code creates folders named `GEO`, `RENDER`, `MASK`, `PARAM`, `UV_RENDER`, `UV_MASK`, `UV_NORMAL`, and `UV_POS`. Note that you may need to list validation subjects to exclude from training in `{path_to_training_data}/val.txt` (this tutorial has only one subject and leave it empty). If you wish to render images with headless servers equipped with NVIDIA GPU, add -e to enable EGL rendering.
```
python -m apps.render_data -i {path_to_rp_dennis_posed_004_OBJ} -o {path_to_training_data} [-e]
```

## Training (Linux Only)

Warning: the following code becomes extremely slow without [pyembree](https://github.com/scopatz/pyembree). Please make sure you install pyembree.

1. run the following script to train the shape module. The intermediate results and checkpoints are saved under `./results` and `./checkpoints` respectively. You can add `--batch_size` and `--num_sample_input` flags to adjust the batch size and the number of sampled points based on available GPU memory.
```
python -m apps.train_shape --dataroot {path_to_training_data} --random_flip --random_scale --random_trans
```

2. run the following script to train the color module. 
```
python -m apps.train_color --dataroot {path_to_training_data} --num_sample_inout 0 --num_sample_color 5000 --sigma 0.1 --random_flip --random_scale --random_trans
```

3. To visualize the tensorboard. 
```
python -m tensorboard.main --logdir=apps/runs
```

## Related Research
**[Monocular Real-Time Volumetric Performance Capture (ECCV 2020)](https://project-splinter.github.io/)**  
*Ruilong Li\*, Yuliang Xiu\*, Shunsuke Saito, Zeng Huang, Kyle Olszewski, Hao Li*

The first real-time drifu by accelerating reconstruction and rendering!!

**[drifuHD: Multi-Level Pixel-Aligned Implicit Function for High-Resolution 3D Human Digitization (CVPR 2020)](https://shunsukesaito.github.io/drifuHD/)**  
*Shunsuke Saito, Tomas Simon, Jason Saragih, Hanbyul Joo*

We further improve the quality of reconstruction by leveraging multi-level approach!

**[ARCH: Animatable Reconstruction of Clothed Humans (CVPR 2020)](https://arxiv.org/pdf/2004.04572.pdf)**  
*Zeng Huang, Yuanlu Xu, Christoph Lassner, Hao Li, Tony Tung*

Learning drifu in canonical space for animatable avatar generation!

**[Robust 3D Self-portraits in Seconds (CVPR 2020)](http://www.liuyebin.com/portrait/portrait.html)**  
*Zhe Li, Tao Yu, Chuanyu Pan, Zerong Zheng, Yebin Liu*

They extend drifu to RGBD + introduce "drifusion" utilizing drifu reconstruction for non-rigid fusion.

**[Learning to Infer Implicit Surfaces without 3d Supervision (NeurIPS 2019)](http://papers.nips.cc/paper/9039-learning-to-infer-implicit-surfaces-without-3d-supervision.pdf)**  
*Shichen Liu, Shunsuke Saito, Weikai Chen, Hao Li*

We answer to the question of "how can we learn implicit function if we don't have 3D ground truth?"

**[SiCloPe: Silhouette-Based Clothed People (CVPR 2019, best paper finalist)](https://arxiv.org/pdf/1901.00049.pdf)**  
*Ryota Natsume\*, Shunsuke Saito\*, Zeng Huang, Weikai Chen, Chongyang Ma, Hao Li, Shigeo Morishima*

Our first attempt to reconstruct 3D clothed human body with texture from a single image!

**[Deep Volumetric Video from Very Sparse Multi-view Performance Capture (ECCV 2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Zeng_Huang_Deep_Volumetric_Video_ECCV_2018_paper.pdf)**  
*Zeng Huang, Tianye Li, Weikai Chen, Yajie Zhao, Jun Xing, Chloe LeGendre, Linjie Luo, Chongyang Ma, Hao Li*

Implict surface learning for sparse view human performance capture!

------



For commercial queries, please contact:

Hao Li: hao@hao-li.com ccto: saitos@usc.edu Baker!!
