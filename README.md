<p align="center">
<h2 align="center">  DeFlow: Self-supervised 3D Motion Estimation of Debris Flow :mountain:</h2>

<p align="center">
    <a href="http://zhuliyuan.net/"><strong>Liyuan Zhu</strong></a>, 
    <a href="https://github.com/yurujaja"><strong>Yuru Jia</strong></a>, 
    <a href="https://shengyuh.github.io/"><strong>Shengyu Huang</strong></a>,
    <a href="https://gseg.igp.ethz.ch/people/scientific-assistance/nicholas-meyer.html"><strong>Nicholas Meyer</strong></a>,
    <a href="https://gseg.igp.ethz.ch/people/group-head/prof-dr--andreas-wieser.html"><strong>Andreas Wieser</strong></a>,
    <a href="https://igp.ethz.ch/personen/person-detail.html?persid=143986"><strong>Konrad Schindler</strong></a>,
    <a href="https://erdw.ethz.ch/en/people/profile.jordan-aaron.html"><strong>Jordan Aaron</strong></a>
  </p>

<p align="center"><strong>ETH Zurich</strong></a>
  <h3 align="center"><a href="https://arxiv.org/abs/">Paper</a> 
  | <a href="">Website</a> | <a href="misc/poster.pdf">Poster</a> | <a href="https://www.research-collection.ethz.ch/handle/20.500.11850/599948">Dataset</a> </h3> 
  <div align="center"></div>



<image src="misc/overview.png"/>
</p>

This repository is the official implementation:

 <b>DeFlow: Self-supervised 3D Motion Estimation of Debris Flow.</b> CVPRW 2023

## Installation :national_park:
First clone our repository:
```bash
git clone https://github.com/Zhu-Liyuan/DeFlow
cd DeFlow
```

You will need to install conda to build the environment.
```bash
conda create -n DeFlow python=3.9
conda activate DeFlow
pip install -r requirements.txt
```

## Dataset and pretrained model
We provide preprocessed debris flow dataset. The preprocessed dataset and checkpoint can be downloaded by running:
```shell
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/DeFlow/DeFlow_Dataset.zip
unzip data.zip
wget --no-check-certificate --show-progress https://share.phys.ethz.ch/~gsg/DeFlow/checkpoint.zip
unzip checkpoint.zip
```
You can also build your own dataset following the structure below
```Shell
├── Data
    ├── Cam1
        ├── 000001.jpg
        ├── 000002.jpg
        .
        .
        ├── 00000X.jpg
    ├── Cam2
        ├── 000001.jpg
        ├── 000002.jpg
        .
        .
        ├── 00000X.jpg
    ├── LiDAR
        ├── 000001.ply
        ├── 000002.ply
        .
        .
        ├── 00000X.ply
├── Transformations
        ├── cam_intrinxics.txt
        ├── LiCam_tranformations.txt
```
To train a model, run:
```bash
python main.py --config_path configs/deflow_default.yaml
```
and you can change the mode to eval in the config file for evaluation.


## Contact
If you have any questions, please let me know: 
- Liyuan Zhu {liyzhu@student.ethz.ch}

## Citation
If you use DeFlow for any academic work, please cite our original paper.
```bibtex
@InProceedings{zhu2023DeFlow,
author = {Liyuan Zhu and Yuru Jia and Shengyu Huang and Nicholas Meyer and Andreas Wieser and Konrad Schindler, Jordan Aaron},
title = {DEFLOW: Self-supervised 3D Motion Estimation of Debris Flow},
booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
month = {June},
year = {2023}
}
```

Additional, we thank the respective developers of the following open-source projects:
- [PCAccumulation](https://github.com/prs-eth/PCAccumulation) 
- [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow) 
- [Self-mono-sf](https://github.com/visinf/self-mono-sf)
