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
  <h3 align="center"><a href="https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Zhu_DeFlow_Self-Supervised_3D_Motion_Estimation_of_Debris_Flow_CVPRW_2023_paper.pdf">Paper</a> 
  | <a href="https://zhuliyuan.net/deflow">Website</a> | <a href="https://www.research-collection.ethz.ch/handle/20.500.11850/599948">Dataset</a> </h3> 
  <div align="center"></div>



<image src="misc/overview.png"/>
</p>

This repository is the official implementation of paper:
<b>DeFlow: Self-supervised 3D Motion Estimation of Debris Flow</b>, CVPRW 2023.

Existing work on scene flow estimation focuses on autonomous driving and mobile robotics, while automated solutions are lacking for motion in nature, such as that exhibited by debris flows. We propose \deflow, a model for 3D motion estimation of debris flows, together with a newly captured dataset. We adopt a novel multi-level sensor fusion architecture and self-supervision to incorporate the inductive biases of the scene. We further adopt a multi-frame temporal processing module to enable flow speed estimation over time. Our model achieves state-of-the-art optical flow and depth estimation on our dataset, and fully automates the motion estimation for debris flows.

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
unzip DeFlow_Dataset.zip -d data
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

Additionally, we thank the respective developers of the following open-source projects:
- [PCAccumulation](https://github.com/prs-eth/PCAccumulation) 
- [CamLiFlow](https://github.com/MCG-NJU/CamLiFlow) 
- [Self-mono-sf](https://github.com/visinf/self-mono-sf)
