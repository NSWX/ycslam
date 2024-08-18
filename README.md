# Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM

### [Paper](https://arxiv.org/pdf/2304.14377.pdf) | [Project Page](https://hengyiwang.github.io/projects/CoSLAM) | [Video](https://hengyiwang.github.io/projects/Co-SLAM/videos/presentation.mp4)

> **Co-SLAM：神经实时 SLAM 的联合坐标和稀疏参数编码** <br />
> [Hengyi Wang](https://hengyiwang.github.io/), [Jingwen Wang](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<br />
> CVPR 2023

<p align="center">
  <a href="">
    <img src="./media/coslam_teaser.gif" alt="Logo" width="80%">
  </a>
</p>



该存储库包含论文《Co-SLAM：神经实时 SLAM 的联合坐标和稀疏参数编码》的代码，这是一种基于联合编码执行实时相机跟踪和密集重建的神经 SLAM 方法。



## Update

- [x] Co-SLAM代码 [2023-5-12]
- [x] 离线 RGB-D 重建的代码[请点击此处](https://github.com/HengyiWang/Hybrid-Surf)。 [2023-5-12]
- [x] 评估策略、性能分析代码[请点击此处](https://github.com/JingwenWang95/neural_slam_eval)。 [2023-5-18]
- [x] 有关参数和使用 iPhone/iPad Pro 创建序列的教程[请点击此处](https://github.com/HengyiWang/Co-SLAM/blob/main/DOCUMENTATION.md)。 [2023-5-26]
- [ ] 使用 RealSense 创建序列的教程

## Installation

请按照以下说明安装存储库和依赖项。

```bash
git clone https://github.com/HengyiWang/Co-SLAM.git
cd Co-SLAM
```



### Install the environment

```bash
# Create conda environment
conda create -n coslam python=3.7
conda activate coslam

# Install the pytorch first (Please check the cuda version)
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install all the dependencies via pip (Note here pytorch3d and tinycudann requires ~10min to build)
pip install -r requirements.txt

# Build extension (marching cubes from neuralRGBD)
cd external/NumpyMarchingCubes
python setup.py install

```



对于tinycudann，如果使用GPU时无法访问网络，也可以尝试从源代码构建，如下所示：

```bash
# Build tinycudann 
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn

# Try this version if you cannot use the latest version of tinycudann
#git reset --hard 91ee479d275d322a65726435040fc20b56b9c991
cd tiny-cuda-nn/bindings/torch
python setup.py install
```



## Dataset

#### Replica

将 iMAP 作者生成的副本数据集序列下载到`./data/Replica`文件夹中。

```bash
bash scripts/download_replica.sh # Released by authors of NICE-SLAM
```



#### ScanNet

请按照[ScanNet](http://www.scan-net.org/)网站上的步骤操作，并使用[代码](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py)从`.sens`文件中提取颜色和深度帧。

#### Synthetic RGB-D dataset

将由neuralRGBD作者生成的合成RGB-D数据集的序列下载到`./data/neural_rgbd_data`文件夹中。我们排除 BundleFusion 生成的具有 NaN 姿势的场景。

```bash
bash scripts/download_rgbd.sh 
```



#### TUM RGB-D

将 TUM RGB-D 数据集的 3 个序列下载到`./data/TUM`文件夹中。

```bash
bash scripts/download_tum.sh 
```



## Run

您可以使用以下代码运行 Co-SLAM：

```
python coslam.py --config './configs/{Dataset}/{scene}.yaml 
```



您可以使用以下代码运行具有多处理功能的 Co-SLAM：

```
python coslam_mp.py --config './configs/{Dataset}/{scene}.yaml 
```



## Evaluation

我们采用稍微不同的评估策略来衡量重建的质量，您可以[在此处](https://github.com/JingwenWang95/neural_slam_eval)找到代码。请注意，如果您想遵循NICE-SLAM的评估协议，请参阅我们的补充材料以了解详细的参数设置。

## Acknowledgement

我们改编了一些很棒的存储库的代码，包括[NICE-SLAM](https://github.com/cvg/nice-slam) 、 [NeuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction) 、 [tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn) 。感谢您提供代码。我们还感谢[NICE-SLAM](https://github.com/cvg/nice-slam)的[Zihan Zhu](https://zzh2000.github.io/)和[iMAP](https://edgarsucar.github.io/iMAP/)的[Edgar Sucar](https://edgarsucar.github.io/)对我们有关其方法细节的询问做出了及时答复。

这里介绍的研究得到了思科研究中心和伦敦大学学院基础人工智能博士培训中心赞助的研究奖的支持，资助号为 EP/S021566/1。该项目利用了由 EPSRC (EP/T022205/1) 资助的 Tier 2 HPC 设施 JADE2 上的时间。

## Citation

如果您发现我们的代码或论文对您的研究有用，请考虑引用：

```
@inproceedings{wang2023coslam,
        title={Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM},
        author={Wang, Hengyi and Wang, Jingwen and Agapito, Lourdes},
        booktitle={CVPR},
        year={2023}
}
```

