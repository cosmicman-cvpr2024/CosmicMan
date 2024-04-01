# CosmicMan: A Text-to-Image Foundation Model for Humans
<img src="./assets/teaser_1118_2 2.pdf" width="96%" height="96%">

[Shikai Li](mailto:lishikai@pjlab.org.cn), [Jianglin Fu](https://github.com/arleneF), [Kaiyuan Liu](), [Wentao Wang](), [Kwan-Yee Lin](https://kwanyeelin.github.io/), [Wayne Wu](https://wywu.github.io/) <br>
**[[Video Demo]](https://www.youtube.com/watch?v=fk2fniU6oyM)** | **[[Project Page]](https://cosmicman-cvpr2024.github.io/)** | **[[Paper]]()**

**Abstract:** *We present **CosmicMan**, a text-to-image foundation model specialized for generating high-fidelity human images. Unlike current general-purpose foundation models that are stuck in the dilemma of inferior quality and text-image misalignment for humans, CosmicMan enables generating photo-realistic human images with meticulous appearance, reasonable structure, and precise text-image alignment with detailed dense descriptions. At the heart of CosmicMan's success are the new reflections and perspectives on data and model: (1) We found that data quality and a scalable data production flow are essential for the final results from trained models. Hence, we propose a new data production paradigm **Annotate Anyone**, which serves as a perpetual data flywheel to produce high-quality data with accurate yet cost-effective annotations over time. Based on this, we constructed a large-scale dataset **CosmicMan-HQ 1.0**, with 6 Million high-quality real-world human images in a mean resolution of 1488x1255, and attached with precise text annotations deriving from 115 Million attributes in diverse granularities. (2) We argue that a text-to-image foundation model specialized for humans must be pragmatic - easy to integrate into down-streaming tasks while effective in producing high-quality human images. Hence, we propose to model the relationship between dense text descriptions and image pixels in a decomposed manner, and present **D**ecomposed-**A**ttention-**R**efocus**ing** (**Daring**). Daring is a training framework that seamlessly decomposes the cross-attention features in the existing text-to-image diffusion model, and enforces attention refocusing without adding extra modules. Through Daring, we show that explicitly discretizing continuous text space into several basic groups that align with human body structure is the key to tackling the misalignment problem in a breeze.* <br>

## Updates
<!-- - [01/01/2024] Pretrained model and inference scripts are released.
- [26/09/2023] Our paper is released on arXiv. -->
- [01/04/2023] Our work has been accepted by CVPR2024!


## TODOs
- [ ] Release technical report.
- [ ] Release Inference code.
- [ ] Release pretrained models.
- [ ] Release training code.

<!-- 
## Usage

### Installation
To work with this project on your own machine, you need to install the environmnet as follows: 

```
conda env create -f environment.yml
conda activate unitedhuman
```

### Pretrained models
Please put the downloaded [pretrained models](https://drive.google.com/file/d/1sgtMRWZJ1v4rVzQUaMNeZ8oGv01CyHqm/view?usp=sharing) under the folder 'models'.

### Inference
This script generates samples with [target_size], you can set it to 256, 512, 1024, 2048.
if [only_mean] is set to true, you will get image generate by mean latent.
```
python inference.py --path_list models/network-snapshot-v1.pkl --only_mean --target_size 2048
``` -->


## Related Work
* (ECCV 2022) **StyleGAN-Human: A Data-Centric Odyssey of Human Generation**, Jianglin Fu et al. [[Paper](https://arxiv.org/pdf/2204.11823.pdf)], [[Project Page](https://stylegan-human.github.io/)], [[Dataset](https://github.com/stylegan-human/StyleGAN-Human)]

## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@inproceedings{cosmicman,
      title = {CosmicMan: A Text-to-Image Foundation Model for Humans},
      author = {Li, Shikai and Fu, Jianglin and Liu, Kaiyuan and Wang, Wentao and Lin, Kwan-Yee and Wu, Wayne},
      booktitle = {Computer Vision and Pattern Recognition (CVPR)},
      year = {2024}
}
```

