# Neural SLAM Evaluation Benchmark
此存储库包含 Co-SLAM： Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM 的评估代码，这是一种基于联合编码执行实时相机跟踪和密集重建的神经 SLAM 方法。

### [Project Page](https://hengyiwang.github.io/projects/CoSLAM) | [Paper](https://arxiv.org/abs/2304.14377)
> Co-SLAM：神经实时 SLAM 的联合坐标和稀疏参数编码 <br />
> [Hengyi Wang*](https://hengyiwang.github.io/), [Jingwen Wang*](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/) <br />
> CVPR 2023

<p align="center">
  <a href="">
    <img src="./media/coslam_teaser.gif" alt="Logo" width="80%">
  </a>
</p>

在此 repo 中，我们还对同一评估协议下现有的开源 RGB-D 神经 SLAM 方法进行了全面比较。我们希望这将有利于神经SLAM领域的研究。

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#installation">Installation</a>
    </li>
    <li>
      <a href="#datasets">Datasets</a>
    </li>
    <li>
      <a href="#evaluation-protocol">Evaluation Protocol</a>
    </li>
    <li>
      <a href="#run-evaluation">Run Evaluation</a>
    </li>
    <li>
      <a href="#benchmark">Benchmark</a>
    </li>
    <li>
      <a href="#acknowledgement">Acknowledgement</a>
    </li>
    <li>
      <a href="#citation">Citation</a>
    </li>
  </ol>
</details>

## Installation
此存储库假设您已经从[Co-SLAM](https://github.com/HengyiWang/Co-SLAM)主存储库配置了环境。然后，您还需要以下依赖项：
* Open3D
* pyglet
* pyrender

您可以通过运行以下命令来安装这些依赖项：

```bash
conda activate coslam
pip install -r requirements.txt
```

## Datasets
在 iMAP 和 NICE-SLAM 之后，我们在 Replica、ScanNet 和 TUM RGB-D 数据集上评估我们的方法。我们对 NeuralRGBD 的合成数据集进行了进一步的实验，该数据集包含许多薄结构并模拟真实深度传感器测量中存在的噪声。我们在[主存储库](https://github.com/HengyiWang/Co-SLAM)中提供了下载链接。

除了这些序列和[NICE-SLAM](https://github.com/cvg/nice-slam/blob/master/scripts/download_apartment.sh)作者捕获的公寓序列之外，我们还使用 RealSense D435i 深度相机收集了我们自己的真实室内场景（MyRoom），该相机在机器人社区中更受欢迎，深度质量稍差比 Azure Kinect 还差。您可以从[这里](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EW-_dVBx8pFImN-nzlL7d9YB9ikr_GMkI339cSFK4lsFWw?e=d1ksWJ)下载 MyRoom 序列（~15G）。

## Evaluation Protocol

### Mesh Culling
正如我们[补充材料](https://arxiv.org/abs/2304.14377)第 1.2 节所述，对于神经 SLAM 方法，由于神经网络的外推能力，需要进行网格剔除来评估重建质量。这种外推特性为神经 SLAM 方法带来了孔洞填充能力，但也可能在感兴趣区域 (ROI) 之外产生不需要的伪影。理想情况下，我们需要一种剔除策略，可以删除 ROI 之外不需要的部分，并保持所有其他部分不变。

神经 SLAM/重建系统中使用的现有剔除方法要么基于***平截头体***（NICE-SLAM 和 iMAP），要么基于***平截头体+遮挡***（Neural-RGBD 和 GO-Surf）策略。第一个可能会在 ROI 之外留下伪影（例如墙后的伪影），而第二个会移除 ROI 内被遮挡的部分。在Co-SLAM中，我们建议进一步使用***平截头体+遮挡+虚拟相机***，引入额外的虚拟视图来覆盖感兴趣区域内的被遮挡部分。请参阅我们[补充材料](https://arxiv.org/abs/2304.14377)的第 1.2 节以获取更多解释和详细信息。

我们提供的剔除脚本包含了上述所有三种剔除策略，以防您想要遵循其他两个协议。下面是一个用法示例：
```bash
INPUT_MESH=output/Replica/office0/demo/mesh_xxxx.ply
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --gt_pose  # Co-SLAM strategy
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --gt_pose  # Neural-RGBD/GO-Surf strategy
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --gt_pose  # iMAP/NICE-SLAM strategy
```

### Command line arguments
- `--config $CONFIG_FILE` 输入网格场景的配置文件
- `--input_mesh $INPUT_MESH` 要剔除的网格的路径
- `--output_mesh $OUTPUT_MESH` 保存剔除网格的路径（可选）
- `--remove_occlusion` 是否移除自闭塞
- `--virt_cam_path` 虚拟摄像机目录的路径（可选）
- `--virtual_cameras` 是否使用虚拟摄像机
- `--gt_pose` 使用或不使用 GT 姿势
- `--ckpt_path` 重建检查点的路径（可选）

请注意，要使用 Co-SLAM 剔除策略，您需要虚拟摄像机视图。

我们提供在Replica和Neural-RGBD数据集上评估Co-SLAM所需的数据。


### Virtual Camera Views
虚拟视图的目的只是覆盖现有视图可能无法观察到的区域，因此它们的选择非常灵活。为了让您了解如何做到这一点，这里我们还提供了一个简单的示例 Python 脚本，用于以交互式方式为 Replica 序列创建虚拟摄像机：
```bash
python create_virtual_cameras_replica.py --config configs/Replica/office0.yaml --data_dir data/Replica/office0
```
它将首先创建具有 GT 姿势的 TSDF-Fusion 网格。创建网格后，将弹出一个 Open3D 窗口，您可以使用鼠标调整视点以覆盖未观察到的区域。按 `.` 键盘上的按钮保存视点。

## Run Evaluation

### Reconstruction
要评估重建质量，请首先下载评估所需的数据：
* [Replica](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EUfNXQ_qps5DtYJP7FNegxQBHoQPUpg63TcVOUzFAubeDQ?e=aGTFrp)
* [SyntheticRGBD](https://liveuclac-my.sharepoint.com/:u:/g/personal/ucabjw4_ucl_ac_uk/EW_AaEdHND1ElCLK0GXLBXgBND4iKy_mKza0Xf8GxQYq5w?e=9N0kTI)

其中包含虚拟摄像机视图、使用我们建议的剔除策略（用于 3D 度量）和 GT 网格上看不见的点（用于 2D 度量）与这些虚拟视图一起剔除的 GT 相机网格。为了实现可重复性，我们还包括了采样的 1000 个相机姿势进行 2D 评估。

```
<scene_name>           
├── virtual_cameras             # virtual cameras
    ├── 0.txt     
    ├── 0.png
    ├── 1.txt
    ├── 1.png
    ...
├── sampled_poses_1000.npz      # sampled 1000 camrera poses
├── gt_pc_unseen.npy            # point cloud of unseen part
├── gt_unseen.ply               # mesh of unseen part
├── gt_mesh_cull_virt_cams.ply  # culled ground-truth mesh
```

然后运行剔除脚本来剔除重建的网格
```python
# Put your own path to reconstructed mesh. Here is just an example
INPUT_MESH=output/Replica/office0/demo/mesh_track1999.ply
VIRT_CAM_PATH=eval_data/Replica/office0/virtual_cameras
python cull_mesh.py --config configs/Replica/office0.yaml --input_mesh $INPUT_MESH --remove_occlusion --virtual_cameras --virt_cam_path $VIRT_CAM_PATH --gt_pose  # Co-SLAM strategy
```
一旦您获得了剔除的重建网格，评估将遵循类似于 iMAP/NICE-SLAM 的流程。
```python
REC_MESH=output/Replica/office0/demo/mesh_track1999_cull_virt_cams.ply
GT_MESH=eval_data/Replica/office0/gt_mesh_cull_virt_cams.ply
python eval_recon.py --rec_mesh $REC_MESH --gt_mesh $GT_MESH --dataset_type Replica -2d -3d
```

### Tracking
我们遵循完全相同的评估协议来评估平均轨迹误差（ATE）。请参阅[NICE-SLAM](https://github.com/cvg/nice-slam/tree/master)或我们的[主要存储库](https://github.com/HengyiWang/Co-SLAM)了解更多详细信息。

## Benchmark
在本节中，我们将比较其他关于重建质量、跟踪精度和性能分析的方法。所有性能分析均在同一计算平台上进行：一台配备 3.60GHz Intel Core i7-12700K CPU 和单个 NVIDIA RTX 3090ti GPU 的台式 PC。为了排除依赖于方法的实现细节（如数据加载、不同的多处理策略）的影响，我们只报告执行跟踪/映射迭代所需的时间和相应的 FPS。我们还报告了在每个部分下的单个数据集页面中处理每个序列所需的总时间。

### Replica

|     Methods      | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ | 
|:----------------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|       iMAP       |     3.62      |      4.93      |          80.51          |          4.64           |        16.8x6         |       44.8x10       |      9.92       |     2.23      |  0.26M  |
|    NICE-SLAM     |     2.37      |      2.64      |          91.13          |          1.90           |        7.8x10         |       82.5x60       |      13.70      |     0.20      |  17.4M  |
|    Vox-Fusion    |     1.88      |      2.56      |          90.93          |          2.91           |        15.8x30        |       46.0x10       |      2.11       |     2.17      |  0.87M  |
|      ESLAM       |     2.18      |      1.75      |          96.46          |          0.94           |         6.9x8         |       18.4x15       |      18.11      |     3.62      |  9.29M  |
|     Co-SLAM      |     2.10      |      2.08      |          93.44          |          1.51           |        5.8x10         |       9.8x10        |      17.24      |     10.20     |  0.26M  |

这里的跟踪/建图FPS表示完整的跟踪/建图优化周期可以运行的速度，因此与系统的实际运行时FPS不对应。对于整个系统运行时间，我们报告在[benchmark/replica](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/replica)中处理整个序列所需的总时间。另请注意，对于 iMAP*、NICE-SLAM 和 Co-SLAM，副本映射大约每 5 帧发生一次。 Vox-Fusion 采用不同的多处理策略，并尽可能频繁地执行映射。

请参阅[基准/副本](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/replica)以了解每个场景的更多详细信息和细分。

### SyntheticRGBD

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|   iMAP*    |     18.29     |     26.41      |          20.73          |          47.22          |        31.0x50        |      49.1x300       |      0.64       |     0.07      |  0.22M  |
| NICE-SLAM  |     5.95      |      5.30      |          77.46          |          6.32           |        12.3x10        |       50.4x60       |      8.13       |     0.33      |  3.11M  |
| Vox-Fusion |     4.10      |      4.81      |          81.78          |          6.13           |        16.6x30        |       46.2x10       |      2.00       |     2.16      |  0.84M  |
|  Co-SLAM   |     2.95      |      2.96      |          86.88          |          3.02           |        6.4x10         |       10.4x10       |      15.63      |     9.62      |  0.26M  |

这里的跟踪/建图FPS表示完整的跟踪/建图优化周期可以运行的速度，因此与系统的实际运行时FPS不对应。对于整个系统运行时间，我们在[benchmark/rgbd](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/rgbd)中报告处理整个序列所需的总时间。所有实验均使用每种方法的复制品进行。

有关每个场景的更多详细信息，请参阅[benchmark/rgbd](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/rgbd) 。

### ScanNet

|  Methods   | ATE↓<br/>[cm] | ATE↓<br/>w/o align<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ |
|:----------:|:-------------:|:---------------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|   iMAP*    |     36.67     |              -              |        30.4x50        |      44.9x300       |      0.66       |     0.07      |  0.2M   |
| NICE-SLAM  |     9.63      |            23.97            |        12.3x50        |      125.3x60       |      1.63       |     0.13      |  10.3M  |
| Vox-Fusion |     8.22      |              -              |        29.4x30        |       85.8x15       |      1.13       |     0.78      |  1.1M   |
|   ESLAM    |     7.42      |              -              |        7.4x30         |       22.4x30       |      4.54       |     1.49      |  10.5M  |
|  Co-SLAM   |     9.37      |            18.01            |        7.8x10         |       20.2x10       |      12.82      |     4.95      |  0.8M   |
|  Co-SLAM†  |     8.75      |              -              |        7.8x20         |       20.2x10       |      6.41       |     4.95      |  0.8M   |

这里的跟踪/建图FPS表示完整的跟踪/建图优化周期可以运行的速度，因此与系统的实际运行时FPS不对应。对于整个系统运行时间，我们报告在[benchmark/scannet](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/scannet)中处理整个序列所需的总时间。另请注意，在 ScanNet 上，iMAP*、NICE-SLAM 和 Co-SLAM 映射大约每 5 帧发生一次。 Vox-Fusion 采用不同的多处理策略，并尽可能频繁地执行映射。

Please refer to [benchmark/scannet](benchmark/scannet) for more details of each scene.

### TUM-RGBD

| Methods   | ATE↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param |
|-----------|:-------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:------:|
| iMAP      |     4.23      |           -           |          -          |        -        |       -       |   -    |
| iMAP*     |     6.10      |       29.6x200        |      44.3x300       |      0.17       |     0.08      |  0.2M  |
| NICE-SLAM |     2.50      |       47.1x200        |      189.2x60       |      0.11       |     0.09      | 101.6M |
| Co-SLAM   |     2.40      |        7.5x10         |       19.0x20       |      13.33      |     2.63      |  1.6M  |
| Co-SLAM†  |     2.17      |        7.5x20         |       19.0x20       |      6.67       |     2.63      |  1.6M  |

这里的跟踪/建图FPS表示完整的跟踪/建图优化周期可以运行的速度，因此与系统的实际运行时FPS不对应。对于整个系统运行时间，我们在[benchmark/tum](https://github.com/JingwenWang95/neural_slam_eval/blob/main/benchmark/tum)中报告处理整个序列所需的总时间。另请注意，在 TUM-RGBD 上，NICE-SLAM 和 iMAP* 映射每帧发生一次，Co-SLAM 映射每 5 帧发生一次。

Please refer to [benchmark/tum](benchmark/tum) for more details of each scene.

## Acknowledgement

该存储库改编了一些很棒的存储库的代码，包括[NICE-SLAM](https://github.com/cvg/nice-slam) 、 [NeuralRGBD](https://github.com/dazinovic/neural-rgbd-surface-reconstruction)和[GO-Surf](https://github.com/JingwenWang95/go-surf) 。感谢您提供代码。我们还感谢[NICE-SLAM](https://github.com/cvg/nice-slam)的[Zihan Zhu](https://zzh2000.github.io/)和[iMAP](https://edgarsucar.github.io/iMAP/)的[Edgar Sucar](https://edgarsucar.github.io/)快速回复了他们的方法细节。

这里介绍的研究得到了思科研究中心和伦敦大学学院基础人工智能博士培训中心赞助的研究奖的支持，资助号为 EP/S021566/1。该项目利用了由 EPSRC (EP/T022205/1) 资助的 Tier 2 HPC 设施 JADE2 上的时间。

## Citation

如果您发现我们的代码/工作对您的研究有用或希望参考基准测试结果，请考虑引用以下内容：

```bibtex
@article{wang2023co-slam,
  title={Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM},
  author={Wang, Hengyi and Wang, Jingwen and Agapito, Lourdes},
  journal={arXiv preprint arXiv:2304.14377},
  year={2023}
}

@inproceedings{wang2022go-surf,
  author={Wang, Jingwen and Bleja, Tymoteusz and Agapito, Lourdes},
  booktitle={2022 International Conference on 3D Vision (3DV)},
  title={GO-Surf: Neural Feature Grid Optimization for Fast, High-Fidelity RGB-D Surface
  Reconstruction},
  year={2022},
  pages = {433-442},
  organization={IEEE}
}
```

## Contact
如有问题和报告错误，请联系[Hengyi Wang](https://github.com/JingwenWang95/neural_slam_eval/blob/main/hengyi.wang.21@ucl.ac.uk)和[Jingwen Wang](mailto:jingwen.wang.17@ucl.ac.uk) 。
