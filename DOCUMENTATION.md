# Co-SLAM Documentation

### [Paper](https://arxiv.org/pdf/2304.14377.pdf) | [Project Page](https://hengyiwang.github.io/projects/CoSLAM) | [Video](https://hengyiwang.github.io/projects/Co-SLAM/videos/presentation.mp4)

> Co-SLAM: Joint Coordinate and Sparse Parametric Encodings for Neural Real-Time SLAM <br />
> [Hengyi Wang](https://hengyiwang.github.io/), [Jingwen Wang](https://jingwenwang95.github.io/), [Lourdes Agapito](http://www0.cs.ucl.ac.uk/staff/L.Agapito/)<br />
> CVPR 2023

这是 Co-SLAM 的文档，其中包含数据捕获、不同超参数的详细信息。



## 使用 iphone/ipad pro 创建您自己的数据集

1. Download [strayscanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) in App Store
2. 录制 RGB-D 视频
3. 输出文件夹应位于`Files/On My iPad/Stray Scanner/`
4. 为数据集创建您自己的配置文件，设置`dataset: 'iphone'` ，检查`./configs/iPhone`了解详细信息。 （注意Stray Scanner给出的内在函数应除以7.5以对齐RGB-D帧）
5. 为特定场景创建配置文件，您可以自己定义场景绑定，或使用提供的`vis_bound.ipynb`来确定场景绑定

注意：深度图的分辨率是（256, 192），有点太小了。神经 SLAM 的相机跟踪在 iPhone 数据集上不会非常鲁棒。建议使用RealSense进行数据采集。对此有任何建议都将受到欢迎。

## Parameters

### Tracking

```yaml
iter: 10 # 用于跟踪的迭代次数
sample: 1024 # 用于跟踪的样本数
pc_samples: 40960 # 使用点云损失进行跟踪的样本数量 (未使用)
lr_rot: 0.001 # 旋转的学习率
lr_trans: 0.001 # 平移的学习率
ignore_edge_W: 20 # 忽略图像边缘 (宽度)
ignore_edge_H: 20 # 忽略图像边缘 (高度)
iter_point: 0 # 使用点云损失进行跟踪的迭代次数 (未使用)
wait_iters: 100 # 如果连续K个迭代没有改善,则停止优化
const_speed: True # 假设恒定速度用于初始化姿态
best: True # 使用损失最小的姿态 / 使用最后的姿态
```



### Mapping

```yaml
sample: 2048 # BA使用的像素数量
first_mesh: True # 保存第一个网格
iters: 10 # BA的迭代次数

对于表示的学习率, 一个有趣的观察是:
如果你设置 lr_embed=0.001, lr_decoder=0.01, 这使得解码器更依赖于坐标编码, 结果会更好的完成。
这适合于室内场景, 但不适合TUM RGB-D...
lr_embed: 0.01 # HashGrid的学习率
lr_decoder: 0.01 # 解码器的学习率

lr_rot: 0.001 # 旋转的学习率
lr_trans: 0.001 # 平移的学习率
keyframe_every: 5 # 每5帧选择一个关键帧
map_every: 5 # 每5帧进行一次BA
n_pixels: 0.05 # 每帧保存的像素数量
first_iters: 500 # 第一帧映射的迭代次数

由于我们进行全局BA, 我们需要确保1) 每次迭代都有来自当前帧的样本, 2) 不要在当前帧上采样太多像素, 这可能会引入偏差, 建议 min_pixels_cur = 0.01 * #samples
optim_cur: False # 对于具有挑战性的场景, 避免在BA期间优化当前帧姿态
min_pixels_cur: 20 # 当前帧的最小采样像素数
map_accum_step: 1 # 用于累积梯度的模型更新步数
pose_accum_step: 5 # 用于累积梯度的姿态更新步数
map_wait_step: 0 # 等待n次迭代后再开始更新模型
filter_depth: False # 是否过滤出离群点
```



### Parametric encoding 参数编码

```yaml
这些是关于神经场景表示中网格编码的超参数的注释说明。
enc: 'HashGrid' # 网格类型, 包括'DenseGrid', 'TiledGrid'等, 详见论文
tcnn_encoding: True # 使用tcnn编码
hash_size: 19 # Hash表大小, 不同数据集可设置不同值
voxel_color: 0.08 # 颜色网格的体素大小 (如果适用)
voxel_sdf: 0.04 # SDF网格的体素大小 (大于10表示体素维度, 即固定分辨率)
oneGrid: True # 仅使用OneGrid
```



### Coordinate encoding 坐标编码

```python
这些是关于神经场景表示中坐标编码的超参数的注释说明。
enc: 'OneBlob' # 坐标编码的类型
n_bins: 16 # OneBlob的bin数量
```



### Decoder 解码器

```yaml
这些是关于神经场景表示中几何特征、SDF和颜色网络的超参数的注释说明。

geo_feat_dim: 15 # 颜色解码器的几何特征维度
hidden_dim: 32 # SDF MLP的隐藏层维度
num_layers: 2 # SDF MLP的层数
num_layers_color: 2 # 颜色MLP的层数
hidden_dim_color: 32 # 颜色MLP的隐藏层维度
tcnn_network: False # 使用tinycudann MLP还是PyTorch MLP
```



### Training 训练

```yaml
这些是关于神经场景表示的损失函数权重和采样参数的注释说明。以下是中文翻译:

rgb_weight: 5.0 # RGB损失权重
depth_weight: 0.1 # 深度损失权重
sdf_weight: 1000 # SDF损失权重 (截断)
fs_weight: 10 # SDF损失权重 (自由空间)
eikonal_weight: 0 # Eikonal损失权重 (未使用)
smooth_weight: 0.001 # 平滑损失权重 (较小, 因为应用于特征)
smooth_pts: 64 # 用于平滑的随机采样网格维度
smooth_vox: 0.1 # 用于平滑的随机采样网格体素大小
smooth_margin: 0.05 # 采样网格边界
#n_samples: 256
n_samples_d: 96 # 渲染使用的采样点数
range_d: 0.25 # 深度引导采样的范围 [-25cm, 25cm]
n_range_d: 21 # 深度引导采样点数
n_importance: 0 # 重要性采样点数
perturb: 1 # 是否随机扰动 (1:True)
white_bkgd: False
trunc: 0.1 # 截断范围 (室内场景为10cm, RUM RGBD为5cm)
rot_rep: 'quat' # 旋转表示 (轴角不支持恒等初始化)
```

