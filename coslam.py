import os
#os.environ['TCNN_CUDA_ARCHITECTURES'] = '86' #?这是设置环境变量，用于指定目标CUDA架构，可能是在使用TCNN（tiny-cuda-nn）库时设置。

# Package imports
import torch
import torch.optim as optim
import numpy as np
import random
import torch.nn.functional as F
import argparse #提供命令行解析工具，可以通过命令行传递参数，用于配置程序运行时的行为。
import shutil   #提供了高级的文件操作，如复制、删除、移动文件或文件夹。
import json     #用于处理JSON数据的解析和生成，通常用于配置文件的读取或保存。
import cv2

from torch.utils.data import DataLoader
from tqdm import tqdm   #提供了一个进度条工具，用于在循环或迭代过程中显示进度，提升代码的可读性和用户体验。

# Local imports
import config
from model.scene_rep import JointEncoding
from model.keyframe import KeyFrameDatabase
from datasets.dataset import get_dataset
from utils import coordinates, extract_mesh, colormap_image
from tools.eval_ate import pose_evaluation
from optimization.utils import at_to_transform_matrix, qt_to_transform_matrix, matrix_to_axis_angle, matrix_to_quaternion


class CoSLAM():
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.dataset = get_dataset(config)  # 根据配置加载数据集，通常包括RGB图像、深度图、相机位姿等。
        self.create_bounds()                # 建图的边界；创建用于场景重建的边界条件，定义地图构建的空间范围。
        self.create_pose_data()             # 初始化数据结构，用于存储SLAM系统中估计的相机位姿以及数据集中提供的地面真值（ground truth）位姿。
        self.get_pose_representation()      # 确定当前数据集中相机位姿的表示形式（例如轴角表示或四元数表示），不同数据集可能有不同的表示方式。例如，TUM数据集使用轴角表示。
        self.keyframeDatabase = self.create_kf_database(config)		# 创建关键帧数据库，通常是从数据集中每隔一定帧数选取关键帧。例如，在TUM的fr1_desk序列中，每5帧选取一个关键帧。
        #!  -------------------- 1. Scene representation: 网络构建  -------------------- 
        # JointEncoding函数内部完成Encoding和Decoding,将 JointEncoding 类实例化后的模型 self.model 移动到 self.device 指定的设备上
        self.model = JointEncoding(config, self.bounding_box).to(self.device)  # 得到encoding/decoding网络，用于获得深度和颜色信息
        #! ----------------------------------------------------------------------------

    def seed_everything(self, seed):    #? 用于设置各种随机数生成器的种子（seed），以确保代码运行的可重复性
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)    #? 确保哈希种子是确定的
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        
    def get_pose_representation(self):
        '''
        决定使用轴角（axis-angle）还是四元数（quaternion）来表示旋转，并设置相应的转换函数
        '''
        if self.config['training']['rot_rep'] == 'axis_angle':
            self.matrix_to_tensor = matrix_to_axis_angle        # 旋转矩阵 -> 轴角
            self.matrix_from_tensor = at_to_transform_matrix    # 轴角 -> 旋转矩阵
            print('Using axis-angle as rotation representation, identity init would cause inf')
        
        elif self.config['training']['rot_rep'] == "quat":
            print("Using quaternion as rotation representation")
            self.matrix_to_tensor = matrix_to_quaternion
            self.matrix_from_tensor = qt_to_transform_matrix
        else:
            raise NotImplementedError
        
    def create_pose_data(self):
        '''
        初始化存储估计的相机位姿（camera-to-world, c2w）的数据结构，并加载相应的地面真值（ground truth, gt）位姿
        '''
        self.est_c2w_data = {}      # 存储估计的绝对相机位姿数据。绝对位姿是相对于世界坐标系的
        self.est_c2w_data_rel = {}  # 存储估计的相对相机位姿数据。相对位姿通常是指当前帧相对于上一帧或某一参考帧的位姿。
        self.load_gt_pose() 
    
    def create_bounds(self):
        '''
        获取场景的预定义边界
        '''
        self.bounding_box = torch.from_numpy(np.array(self.config['mapping']['bound'])).to(torch.float32).to(self.device)
        self.marching_cube_bound = torch.from_numpy(np.array(self.config['mapping']['marching_cubes_bound'])).to(torch.float32).to(self.device)

    def create_kf_database(self, config):  
        '''
        创建关键帧数据库
        '''
        num_kf = int(self.dataset.num_frames // self.config['mapping']['keyframe_every'] + 1)  #! +1 是为了确保在边界情况下也至少有一个关键帧
        print('#kf:', num_kf)
        print('#Pixels to save:', self.dataset.num_rays_to_save)
        return KeyFrameDatabase(config, 
                                self.dataset.H, 
                                self.dataset.W, 
                                num_kf, 
                                self.dataset.num_rays_to_save, 
                                self.device)
    
    def load_gt_pose(self):
        '''
        加载地面真实位姿（ground truth poses），并将它们存储在一个字典 self.pose_gt 中
        '''
        self.pose_gt = {}
        for i, pose in enumerate(self.dataset.poses):   #? enumerate(iterable, start=0)是一个内置函数，返回一个包含索引和元素的元组 (index, element)
            self.pose_gt[i] = pose
 
    def save_state_dict(self, save_path):   # 将模型的状态字典（包括所有的模型参数，例如权重和偏置）保存到指定路径。这个文件会以 .pth 或 .pt 作为扩展名
        torch.save(self.model.state_dict(), save_path)
    
    def load(self, load_path):              # 从指定路径加载模型的状态字典，并将这些参数加载到模型中。
        self.model.load_state_dict(torch.load(load_path))
    
    def save_ckpt(self, save_path):
        """
        保存模型参数和估计位姿到指定路径。
        Args:
            save_path (str): 要保存的检查点文件路径。
        Returns:
            None
        """
        # 创建一个字典，其中包含当前的模型参数和估计的位姿数据
        save_dict = {
            'pose': self.est_c2w_data,          # 估计的绝对相机姿态
            'pose_rel': self.est_c2w_data_rel,  # 估计的相对相机姿态
            'model': self.model.state_dict()    # 模型的状态字典
        }
        # 将字典保存到指定的路径
        torch.save(save_dict, save_path)
        # 打印一条信息以确认检查点已保存
        print('Save the checkpoint')


    def load_ckpt(self, load_path):
        """
        从指定路径加载模型参数和估计姿态。
        Args:
            load_path (str): 要加载的检查点文件路径。

        Returns:
            None
        """
        # 从指定路径加载保存的模型和姿态数据
        dict = torch.load(load_path)
        # 加载模型的状态字典
        self.model.load_state_dict(dict['model'])
        # 加载估计的绝对相机姿态
        self.est_c2w_data = dict['pose']
        # 加载估计的相对相机姿态
        self.est_c2w_data_rel = dict['pose_rel']


    def select_samples(self, H, W, samples):
        '''
        从图像中随机选择样本
        '''
        #indice = torch.randint(H*W, (samples,))
        indice = random.sample(range(H * W), int(samples)) #! 生成一个从 0 到 H*W - 1 的整数序列, 随机选择 samples 个不重复样本
        indice = torch.tensor(indice)
        return indice

    def get_loss_from_ret(self, ret, rgb=True, sdf=True, depth=True, fs=True, smooth=False):
        '''
        获得训练损失
        参数：
            ret: 一个字典，包含了各个损失项的值，如 rgb_loss、depth_loss、sdf_loss 和 fs_loss。
            rgb, sdf, depth, fs: 布尔值，用于指示是否计算相应的损失项。
            smooth: 布尔值，指示是否计算平滑损失。
        '''
        loss = 0 # 初始化总损失为 0
        if rgb:
            loss += self.config['training']['rgb_weight'] * ret['rgb_loss']
        if depth:
            loss += self.config['training']['depth_weight'] * ret['depth_loss']
        if sdf:
            loss += self.config['training']['sdf_weight'] * ret["sdf_loss"]
        if fs:
            loss +=  self.config['training']['fs_weight'] * ret["fs_loss"]
        
        if smooth and self.config['training']['smooth_weight']>0:
            loss += self.config['training']['smooth_weight'] * self.smoothness(
                                                                                self.config['training']['smooth_pts'], # 平滑损失中的点数
                                                                                self.config['training']['smooth_vox'], # 平滑损失中的体素数
                                                                                margin=self.config['training']['smooth_margin']) # 平滑损失的边际值
        
        return loss             

    def first_frame_mapping(self, batch, n_iters=100):
        '''
        处理第一帧的映射操作
        Params:
            batch['c2w']: [1, 4, 4] - 相机姿态矩阵
            batch['rgb']: [1, H, W, 3] - RGB颜色图像
            batch['depth']: [1, H, W, 1] - 深度图像
            batch['direction']: [1, H, W, 3] - 射线方向
            n_iters: 表示优化迭代次数的整数，默认为100。
        Returns:
            ret: dict - 包含模型前向计算结果的字典
            loss: float - 训练损失值的浮点数
        
        '''
        # 1.开始映射并初始化变量：
        print('First frame mapping...')
        c2w = batch['c2w'][0].to(self.device)   # 从 batch 中提取并传送相机姿态矩阵到指定设备（如GPU）
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        # 2.将模型设置为训练模式
        self.model.train()  # 将模型设置为训练模式，使得在前向传播时启用dropout和batchnorm等操作

        # 3.训练循环（共执行 n_iters 次）：
        for i in range(n_iters):
            # 此处的循环与 tracking_render 代码中的2.2部分for循环代码、和 BA 的for循环代码是比较相似的
            #? 1问.哪里相似？
            # 1答：射线的选择和构建很相似，self.select_samples像素随机选择，根据原点rays_o和方向rays_d构建射线
            #? 2问：为什么相似？
            # 2答：因为都要使用Ray Sampling，我们这里扩展 2.1 3.2 Ray Sampling的范畴，纯粹的光线采样是定义在forward里的render_rays里的，但这里前置的步骤————像素的选择，射线构建，目标颜色值和深度值的获取，都是必不可少的
            self.map_optimizer.zero_grad()  # 清除当前梯度，准备计算新的梯度。
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample']) # 从给定的图像中随机选择像素点样本。
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H) # 将选取的索引转换为图像中的行和列坐标
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device) # 从 batch 中获取这些像素点对应的射线方向 direction，并传输到设备上
            #? squeeze(0) 去除维度为1的批次维度（从 [1, H, W, 3] 变为 [H, W, 3]）。
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device) # 获取这些像素点的目标颜色值。
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1) # 获取这些像素点的目标深度值
            #? unsqueeze(-1) 在最后一维添加一个维度，使其从 [H, W] 变为 [H, W, 1]

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1) # 获取射线的原点。
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1) # 计算射线在世界坐标系中的方向。

            # Forward
            # Sec 3.2 Ray Sampling 藏在self.model.forward里
            ret = self.model.forward(rays_o, rays_d, target_s, target_d) # 调用模型前向传播函数，计算预测结果
            loss = self.get_loss_from_ret(ret) # 从前向传播结果中计算损失。
            loss.backward() # 通过反向传播计算损失的梯度。
            self.map_optimizer.step() # 调用优化器的 step 方法，更新模型参数。
        
        # 4.关键帧的处理：
        # 第一帧将始终是关键帧
        # 第一帧很重要，肯定是关键帧，在NICE_SLAM里提到，第一帧来建立一个参考点，用于建立后续帧的相对位置和方向
        self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth']) # 将第一帧作为关键帧加入数据库，并根据配置参数决定是否过滤深度信息
        if self.config['mapping']['first_mesh']:
            self.save_mesh(0)
        
        # 5.结束映射：
        print('First frame mapping done')
        return ret, loss

    
    def current_frame_mapping(self, batch, cur_frame_id):
        '''
        处理当前帧的映射操作
        根据if判断，函数很早就会return，没有真正执行
        与第一帧映射不同，这里的相机姿态不是从数据中获取，而是从先前估计的数据中提取的
        Params:
            batch['c2w']: [1, 4, 4]
            batch['rgb']: [1, H, W, 3]
            batch['depth']: [1, H, W, 1]
            batch['direction']: [1, H, W, 3]
            cur_frame_id：当前帧的标识符，用于从已估计的相机姿态中获取对应的姿态矩阵。
        Returns:
            ret: dict
            loss: float
        
        '''
        #! 提前终止条件
        # yaml里, ['mapping']['cur_frame_iters'] = 0，return，不再执行
        if self.config['mapping']['cur_frame_iters'] <= 0:
            return
        
        #! 打印提示信息，表示当前帧映射过程开始。
        print('Current frame mapping...')
        
        c2w = self.est_c2w_data[cur_frame_id].to(self.device)

        self.model.train()

        # Training
        for i in range(self.config['mapping']['cur_frame_iters']):
            self.cur_map_optimizer.zero_grad()
            indice = self.select_samples(self.dataset.H, self.dataset.W, self.config['mapping']['sample'])
            
            indice_h, indice_w = indice % (self.dataset.H), indice // (self.dataset.H)
            rays_d_cam = batch['direction'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'].squeeze(0)[indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[indice_h, indice_w].to(self.device).unsqueeze(-1)

            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Sec 3.2 Ray Sampling 藏在self.model.forward里
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)
            loss = self.get_loss_from_ret(ret)
            loss.backward()
            self.cur_map_optimizer.step()
        
        
        return ret, loss

    def smoothness(self, sample_points=256, voxel_size=0.1, margin=0.05, color=False):
        '''
        计算特征网格（voxel grid）的平滑度损失，以确保生成的3D网格在空间上保持平滑和连续性。
        通过对网格中相邻采样点之间的SDF（Signed Distance Function）值进行差值计算，该函数评估网格的平滑度，并返回一个损失值。
        参数: 
            sample_points (int, 默认值: 256):用于生成网格的采样点数。
                该参数决定在每个维度上生成多少个采样点，从而确定整体网格的密度。
            voxel_size (float, 默认值: 0.1):每个体素（voxel）的大小。
                体素是3D网格中的最小单位，该参数用于控制采样点之间的间隔。
            margin (float, 默认值: 0.05):网格内的边距，用于防止采样点落在包围盒的边界上。
                这个参数确保采样点在网格内有适当的偏移量。
            color (bool, 默认值: False):是否考虑颜色信息。
                此参数在当前函数中未使用，但可以用于扩展功能，以支持基于颜色信息的平滑度计算。
        返回值: 
            loss (torch.Tensor):平滑度损失值。
                这个标量值表示网格在空间上连续性的好坏。值越小，表示网格越平滑，变化越平缓；值越大，表示网格中存在较大的不连续性。
        '''
        volume = self.bounding_box[:, 1] - self.bounding_box[:, 0] # 计算包围盒的体积

        grid_size = (sample_points-1) * voxel_size  # 计算网格的大小
        offset_max = self.bounding_box[:, 1]-self.bounding_box[:, 0] - grid_size - 2 * margin # 计算偏移的最大值

        offset = torch.rand(3).to(offset_max) * offset_max + margin # 生成一个随机偏移量 offset，以便在网格内随机定位采样点。这确保了采样点不会总是位于相同的位置。
        coords = coordinates(sample_points - 1, 'cpu', flatten=False).float().to(volume) # 生成网格坐标
        pts = (coords + torch.rand((1,1,1,3)).to(volume)) * voxel_size + self.bounding_box[:, 0] + offset # 计算实际的采样点

        if self.config['grid']['tcnn_encoding']: #? 检查配置是否启用了 tcnn_encoding，如果启用，则需要对采样点进行归一化处理。
            pts_tcnn = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
        

        sdf = self.model.query_sdf(pts_tcnn, embed=True) # 查询SDF（Signed Distance Function）值
        tv_x = torch.pow(sdf[1:,...]-sdf[:-1,...], 2).sum() # 计算X轴方向上的总变差（TV）。通过相邻点之间的SDF值差的平方和来计算。
        tv_y = torch.pow(sdf[:,1:,...]-sdf[:,:-1,...], 2).sum() # 计算Y轴方向上的总变差（TV）
        tv_z = torch.pow(sdf[:,:,1:,...]-sdf[:,:,:-1,...], 2).sum()

        loss = (tv_x + tv_y + tv_z)/ (sample_points**3) # 计算最终的平滑度损失 loss。将三个方向上的总变差相加，然后除以采样点的数量，以标准化损失值。

        return loss
    
    def get_pose_param_optim(self, poses, mapping=True):
        """
        创建用于优化姿态参数的优化器（Optimizer）。

        Args:
            poses (torch.Tensor): 输入的姿态矩阵，形状为 [N, 4, 4]，其中 N 是姿态的数量。
                                每个姿态矩阵的形式为 4x4，包含旋转矩阵和位移向量。
            mapping (bool): 一个布尔值，指示当前任务是否是映射（mapping）。
                            如果为True，则使用'mapping'任务的学习率; 如果为False，则使用'tracking'任务的学习率。

        Returns:
            cur_rot (torch.nn.parameter.Parameter): 当前优化的旋转矩阵参数，形状为 [N, 3, 3]。
            cur_trans (torch.nn.parameter.Parameter): 当前优化的平移向量参数，形状为 [N, 3]。
            pose_optimizer (torch.optim.Adam): 优化器对象，用于优化旋转和位移参数。
        """
        # 确定当前任务是映射（mapping）还是跟踪（tracking），并选择对应的任务标识符
        task = 'mapping' if mapping else 'tracking'

        # 从姿态矩阵中提取平移向量（位置），并将其转换为可训练的参数
        cur_trans = torch.nn.parameter.Parameter(poses[:, :3, 3])

        # 从姿态矩阵中提取旋转矩阵，并将其转换为可训练的参数
        # 这里假设self.matrix_to_tensor方法将旋转矩阵转换为张量（例如四元数或旋转向量形式）
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(poses[:, :3, :3]))

        # 使用Adam优化器来优化旋转矩阵和平移向量
        # 旋转和位移的学习率分别使用self.config[task]['lr_rot']和self.config[task]['lr_trans']
        pose_optimizer = torch.optim.Adam([
            {"params": cur_rot, "lr": self.config[task]['lr_rot']},
            {"params": cur_trans, "lr": self.config[task]['lr_trans']}
        ])

        # 返回旋转矩阵参数、平移向量参数，以及优化器对象
        return cur_rot, cur_trans, pose_optimizer
    
    def global_BA(self, batch, cur_frame_id):
        """
        进行全局BA（Bundle Adjustment），对所有关键帧和当前帧进行优化。

        Args:
            batch (dict): 包含以下键值的字典：
                - 'c2w' (torch.Tensor): ground truth相机姿态，形状为 [1, 4, 4]。
                - 'rgb' (torch.Tensor): RGB图像，形状为 [1, H, W, 3]。
                - 'depth' (torch.Tensor): 深度图像，形状为 [1, H, W, 1]。
                - 'direction' (torch.Tensor): 视角方向，形状为 [1, H, W, 3]。
            cur_frame_id (int): 当前帧的ID。
        Returns:
            None

        """
        pose_optimizer = None # 初始化姿态优化器

        # 提取所有关键帧（KF）的姿态和对应的帧ID
        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # frame ids for all KFs, used for update poses after optimization
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframeDatabase.frame_ids) < 2:
            #  如果关键帧数据库中的帧数少于2，则不优化姿态，直接使用这些帧的姿态
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            # 如果有两个或更多关键帧，将第一个关键帧的姿态固定，并对其他关键帧姿态进行优化
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            # optim_cur符号位判断，代表是否优化当前帧
            # 如果优化当前帧，在get_pose_param_optim()的传参里加入当前帧，poses_all里不重复加入了
            # 如果不优化当前帧，get_pose_param_optim()的传参无当前帧，poses_all里加入当前帧
            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.get_pose_param_optim(poses[1:])
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # 设置优化器
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()

        # 获取当前帧的光线数据，包括方向、目标RGB值和深度值，并展平
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        # 进行迭代优化
        # tum.yaml里，['mapping']['iters'] = 20 , ['mapping']['sample'] = 2048
        for i in range(self.config['mapping']['iters']):

            # 使用真实帧 ID 对光线进行采样
            # rays [bs, 7]
            # frame_ids [bs]

            # 从关键帧数据库中全局采样光线，暗藏玄机，内部对应论文的Contribution II
            rays, ids = self.keyframeDatabase.sample_global_rays(self.config['mapping']['sample'])

            # 从当前帧中随机采样一定数量的光线。采样数量取决于配置的最小像素数和关键帧的数量
            idx_cur = random.sample(range(0, self.dataset.H * self.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframeDatabase.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            # 将关键帧的光线(rays)和当前帧的光线(current_rays_batch)合并
            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)

            # 将光线的方向、目标RGB值和目标深度值提取出来，用新变量赋值
            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # 计算光线的方向和起点
            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 调用模型的前向计算，并计算损失
            # 熟悉的self.model.forward，想必大家看到这里已经不必多说，看到这就知道渲染和计算loss的一整套流程
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            loss = self.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)

            # 更新映射优化器
            # *.yaml里，["mapping"]["map_accum_step"] = 1，(i+1)对1取余始终是0，执行
            if (i + 1) % cfg["mapping"]["map_accum_step"] == 0:
                if (i + 1) > cfg["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            # tum.yaml里，["mapping"]["pose_accum_step"] = 5

            # 更新姿态优化器
            # 提问：for循环内部和外部，分别有一个pose_optimizer的操作，他们有何区别？
            # 答part1：先看下方的循环内部的姿态优化器的更新，姿态优化器在每个 pose_accum_step(5步)之后进行一次更新。这意味着每经过指定的迭代次数，就会更新一次姿态参数。
            if pose_optimizer is not None and (i + 1) % cfg["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # get SE3 poses to do forward pass
                # 计算新的姿态矩阵，并将其转移到计算设备上
                pose_optim = self.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)

                # So current pose is always unchanged
                # 这是作者给出的注释，理解了这段话你就理解了这个姿态优化过程，此理解过程交给同学们
                if self.config['mapping']['optim_cur']:
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # zero_grad here
                pose_optimizer.zero_grad()
        
        # 答part2：在循环结束后，如果存在姿态优化器且有多于一个帧的姿态数据时，进行更新
        # 在优化过程结束后，更新所有关键帧的姿态
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            # 更新所有关键帧的姿态，这是在整个优化过程结束后对关键帧姿态进行最终调整
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            # 如果配置了优化当前帧，将最终的姿态估计应用于当前帧
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
        # 对循环内外中的pose_optimizer的总结：
        # 循环内部的更新是一个逐步的优化过程，用于持续调整姿态估计。
        #   这些更新有助于在整个映射过程中不断改进姿态估计 
        # 循环外的更新是在整个映射过程结束后进行的最终调整
        #   它确保了所有关键帧和当前帧的姿态估计都是最新和最准确的


    #! ********************* 根据论文公式(10)估算当前帧的初始化位姿 *********************   
    def predict_current_pose(self, frame_id, constant_speed=True):
        """
        使用相机运动模型预测当前帧的姿态。

        Args:
            frame_id (int): 当前帧的ID。
            constant_speed (bool): 是否假设相机以恒定速度运动。如果为True，使用前两帧的姿态预测当前帧姿态，否则使用上一帧的姿态。

        Returns:
            torch.Tensor: 预测的当前帧姿态矩阵 [4x4]。
        """
        # 特殊处理：如果是第一帧或未启用恒速模型，直接使用上一帧的姿态作为当前帧的预测姿态
        if frame_id == 1 or (not constant_speed):
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            self.est_c2w_data[frame_id] = c2w_est_prev

        # 恒速运动模型：使用前两帧的相机姿态来预测当前帧的姿态
        else:
            c2w_est_prev_prev = self.est_c2w_data[frame_id - 2].to(self.device)
            c2w_est_prev = self.est_c2w_data[frame_id - 1].to(self.device)
            
            # 估算前前帧和前帧之间的变换矩阵(delta)，将该变换应用到前帧姿态上得到当前帧的预测姿态
            delta = c2w_est_prev @ c2w_est_prev_prev.float().inverse()
            self.est_c2w_data[frame_id] = delta @ c2w_est_prev

        # 返回预测的当前帧姿态矩阵
        return self.est_c2w_data[frame_id]


    def tracking_pc(self, batch, frame_id):
        """
        使用点云损失追踪当前帧的相机姿态。
        （该方法未在论文中使用，但可能在某些情况下有用。）

        Args:
            batch (dict): 包含输入数据的批次，包括相机姿态、RGB图像、深度图像和视角方向。
                - 'c2w' (torch.Tensor): 当前帧的真实相机姿态 [1, 4, 4]。
                - 'rgb' (torch.Tensor): 当前帧的RGB图像 [1, H, W, 3]。
                - 'depth' (torch.Tensor): 当前帧的深度图像 [1, H, W, 1]。
                - 'direction' (torch.Tensor): 视角方向 [1, H, W, 3]。
            frame_id (int): 当前帧的ID。
        """
        
        # 获取真实的相机姿态并转换到计算设备
        c2w_gt = batch['c2w'][0].to(self.device)

        # 预测当前帧的相机姿态
        cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        # 将平移和旋转参数转换为可优化的参数
        cur_trans = torch.nn.parameter.Parameter(cur_c2w[..., :3, 3].unsqueeze(0))
        cur_rot = torch.nn.parameter.Parameter(self.matrix_to_tensor(cur_c2w[..., :3, :3]).unsqueeze(0))

        # 使用Adam优化器对姿态进行优化
        pose_optimizer = torch.optim.Adam([
            {"params": cur_rot, "lr": self.config['tracking']['lr_rot']},
            {"params": cur_trans, "lr": self.config['tracking']['lr_trans']}
        ])
        
        best_sdf_loss = None

        # 忽略图像边缘的宽度和高度
        iW = self.config['tracking']['ignore_edge_W']
        iH = self.config['tracking']['ignore_edge_H']

        thresh = 0

        # 如果迭代次数大于0，开始姿态优化
        if self.config['tracking']['iter_point'] > 0:
            # 从有效区域中采样点云
            indice_pc = self.select_samples(self.dataset.H - iH * 2, self.dataset.W - iW * 2, self.config['tracking']['pc_samples'])
            rays_d_cam = batch['direction'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_s = batch['rgb'][:, iH:-iH, iW:-iW].reshape(-1, 3)[indice_pc].to(self.device)
            target_d = batch['depth'][:, iH:-iH, iW:-iW].reshape(-1, 1)[indice_pc].to(self.device)

            # 过滤掉无效的深度值
            valid_depth_mask = ((target_d > 0.) * (target_d < 5.)).squeeze(1)

            rays_d_cam = rays_d_cam[valid_depth_mask]
            target_s = target_s[valid_depth_mask]
            target_d = target_d[valid_depth_mask]

            # 姿态优化迭代
            for i in range(self.config['tracking']['iter_point']):
                pose_optimizer.zero_grad()
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                # 计算光线的原点和方向
                rays_o = c2w_est[..., :3, -1].repeat(len(rays_d_cam), 1)
                rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)
                pts = rays_o + target_d * rays_d

                # 归一化点云坐标
                pts_flat = (pts - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])

                # 查询颜色和SDF值
                out = self.model.query_color_sdf(pts_flat)

                sdf = out[:, -1]
                rgb = torch.sigmoid(out[:, :3])

                # 计算损失
                loss = 5 * torch.mean(torch.square(rgb - target_s)) + 1000 * torch.mean(torch.square(sdf))

                if best_sdf_loss is None:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()

                # 使用当前姿态更新最优损失
                with torch.no_grad():
                    c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                    if loss.cpu().item() < best_sdf_loss:
                        best_sdf_loss = loss.cpu().item()
                        best_c2w_est = c2w_est.detach()
                        thresh = 0
                    else:
                        thresh += 1

                # 当超过指定迭代次数后停止优化
                if thresh > self.config['tracking']['wait_iters']:
                    break

                loss.backward()
                pose_optimizer.step()

        # 更新姿态估计
        if self.config['tracking']['best']:
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        # 如果不是关键帧，更新相对姿态
        if frame_id % self.config['mapping']['keyframe_every'] != 0:
            kf_id = frame_id // self.config['mapping']['keyframe_every']
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
            c2w_key = self.est_c2w_data[kf_frame_id]
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse()
            self.est_c2w_data_rel[frame_id] = delta

        # 输出最佳损失和相机位姿的损失
        print(f'Best loss: {F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item()}, '
            f'Camera loss: {F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()}')

    
    def tracking_render(self, batch, frame_id):
        '''
        使用当前帧跟踪摄像机姿势。

        Args:
            batch (dict): 包含当前帧的相关数据，包括相机姿态、RGB图像、深度图像和视角方向。
                - 'c2w' (torch.Tensor): 当前帧的真实相机姿态 [B, 4, 4]。
                - 'rgb' (torch.Tensor): RGB 图像 [B, H, W, 3]。
                - 'depth' (torch.Tensor): 深度图像 [B, H, W, 1]。
                - 'direction' (torch.Tensor): 光线方向 [B, H, W, 3]。
            frame_id (int): 当前帧的ID。
        '''

        c2w_gt = batch['c2w'][0].to(self.device) # 从数据集得到当前帧的位姿真值 [4, 4]

        # 初始化当前帧的位姿估计
        if self.config['tracking']['iter_point'] > 0: # 检查是否需要迭代优化（通常不会用到），tum.yaml里['tracking']['iter_point']=0
            cur_c2w = self.est_c2w_data[frame_id]
        else:
            # 本论文采用此方法:  默认方法--使用恒速模型预测当前帧的姿态
            cur_c2w = self.predict_current_pose(frame_id, self.config['tracking']['const_speed'])

        # 初始化用于姿态优化的变量
        indice = None
        best_sdf_loss = None
        thresh=0

        iW = self.config['tracking']['ignore_edge_W'] # 忽略图像边缘宽度
        iH = self.config['tracking']['ignore_edge_H'] # 忽略图像边缘高度

        # 优化准备，创建可优化的旋转 (cur_rot) 和平移 (cur_trans) 参数，以及相应的优化器 (pose_optimizer)
        cur_rot, cur_trans, pose_optimizer = self.get_pose_param_optim(cur_c2w[None,...], mapping=False)

        #! -------------------- Sec 2.2 Camera Tracking --------------------
        # Camera Tracking 是一个多次循环迭代的过程，优化相机姿态以减少渲染结果和实际图像之间的差异
        for i in range(self.config['tracking']['iter']):
            pose_optimizer.zero_grad() # 清零梯度
            c2w_est = self.matrix_from_tensor(cur_rot, cur_trans) # 从优化的参数生成当前的姿态估计

            # Note here we fix the sampled points for optimisation
            # # 固定采样的点用于优化
            if indice is None:
                indice = self.select_samples(self.dataset.H-iH*2, self.dataset.W-iW*2, self.config['tracking']['sample'])
            
                # Slicing  获取采样点对应的光线方向
                indice_h, indice_w = indice % (self.dataset.H - iH * 2), indice // (self.dataset.H - iH * 2)
                rays_d_cam = batch['direction'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            # 目标的RGB和深度值
            target_s = batch['rgb'].squeeze(0)[iH:-iH, iW:-iW, :][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'].squeeze(0)[iH:-iH, iW:-iW][indice_h, indice_w].to(self.device).unsqueeze(-1)

            # 计算光线的原点和方向
            rays_o = c2w_est[...,:3, -1].repeat(self.config['tracking']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:, :3, :3], -1)

            #! -------------------- Sec 2.1 Ray Sampling 环节在self.model.forward()函数内部的render_rays()函数中执行  --------------------
            ret = self.model.forward(rays_o, rays_d, target_s, target_d) ## 渲染光线并计算损失
            # 注意这个函数get_loss_from_ret()，内部将rgb损失，深度损失，sdf损失，fs损失都使用上了
            loss = self.get_loss_from_ret(ret)
            
            # 更新最优损失
            if best_sdf_loss is None:
                best_sdf_loss = loss.cpu().item()
                best_c2w_est = c2w_est.detach()

            with torch.no_grad():
                c2w_est = self.matrix_from_tensor(cur_rot, cur_trans)

                # 如果新的姿态估计比之前的更好，更新最佳估计best_c2w_est
                if loss.cpu().item() < best_sdf_loss:
                    best_sdf_loss = loss.cpu().item()
                    best_c2w_est = c2w_est.detach()
                    thresh = 0
                else:
                    thresh +=1
            
            # 如果连续若干次迭代损失未降低，则提前结束优化
            if thresh >self.config['tracking']['wait_iters']:
                break

            # 反向传播和优化
            loss.backward()
            pose_optimizer.step()
        '''
        选择是否使用最小损失的姿态或最后一次迭代的姿态
        tum.yaml
        tracking best: False      没有选最小的，个人猜测可能是出于实时性的考虑
        '''
        if self.config['tracking']['best']:
            # Use the pose with smallest loss 选最小loss的pose
            self.est_c2w_data[frame_id] = best_c2w_est.detach().clone()[0]
        else:
            # Use the pose after the last iteration 选最后一次迭代得到的pose
            self.est_c2w_data[frame_id] = c2w_est.detach().clone()[0]

        # 对于非关键帧，计算并保存相对于最近关键帧的姿态变换
        if frame_id % self.config['mapping']['keyframe_every'] != 0:        # 如果不是关键帧
            kf_id = frame_id // self.config['mapping']['keyframe_every']    # 前帧所属的关键帧的索引，比如11//5=2 第11帧属于第2个的关键帧
            kf_frame_id = kf_id * self.config['mapping']['keyframe_every']  # 关键帧id，比如2*5，第2个关键帧就是第10帧
            c2w_key = self.est_c2w_data[kf_frame_id]                        # 关键帧的估计位姿
            delta = self.est_c2w_data[frame_id] @ c2w_key.float().inverse() # 当前帧的T 乘上 关键帧的T^-1 得到两帧位姿之间的差异
            self.est_c2w_data_rel[frame_id] = delta                         # 保存差异
        
        # 输出最佳损失和最后一次姿态估计的损失
        print('Best loss: {}, Last loss{}'.format(F.l1_loss(best_c2w_est.to(self.device)[0,:3], c2w_gt[:3]).cpu().item(), F.l1_loss(c2w_est[0,:3], c2w_gt[:3]).cpu().item()))
    
    def convert_relative_pose(self):
        '''
        将相对姿态转换为绝对姿态。
        
        Returns:
            poses (dict): 包含每一帧的绝对姿态估计的字典。
        '''
        poses = {}  # 创建一个空字典，用于存储绝对姿态
        
        # 遍历所有帧，计算并存储绝对姿态
        for i in range(len(self.est_c2w_data)):
            # 如果当前帧是关键帧，直接使用其绝对姿态
            if i % self.config['mapping']['keyframe_every'] == 0:
                poses[i] = self.est_c2w_data[i]
            else:
                # 如果当前帧不是关键帧，则计算其相对于最近关键帧的姿态
                kf_id = i // self.config['mapping']['keyframe_every']  # 找到当前帧所属的关键帧ID
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']  # 计算关键帧的帧ID
                c2w_key = self.est_c2w_data[kf_frame_id]  # 获取关键帧的绝对姿态
                delta = self.est_c2w_data_rel[i]  # 获取当前帧相对于关键帧的相对姿态
                poses[i] = delta @ c2w_key  # 计算当前帧的绝对姿态并存储
        
        return poses  # 返回包含所有帧的绝对姿态的字典
 
    def create_optimizer(self):
        '''
            为映射创建优化器
        '''
        # BA（Bundle Adjustment）优化器：用于优化模型的解码器参数和嵌入函数参数
        trainable_parameters = [{'params': self.model.decoder.parameters(), 'weight_decay': 1e-6, 'lr': self.config['mapping']['lr_decoder']},
                                {'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
        
        # 如果配置中指定了不使用单个网格（'oneGrid' 为 False），则还需优化颜色嵌入函数的参数
        if not self.config['grid']['oneGrid']:
            trainable_parameters.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
        
        # 创建用于整体映射的Adam优化器
        self.map_optimizer = optim.Adam(trainable_parameters, betas=(0.9, 0.99))
        
        # 如果配置中指定了对当前帧进行映射优化（'cur_frame_iters' > 0），则创建另一个优化器
        if self.config['mapping']['cur_frame_iters'] > 0:
            params_cur_mapping = [{'params': self.model.embed_fn.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed']}]
            
            # 如果不使用单个网格，还需优化当前帧的颜色嵌入函数参数
            if not self.config['grid']['oneGrid']:
                params_cur_mapping.append({'params': self.model.embed_fn_color.parameters(), 'eps': 1e-15, 'lr': self.config['mapping']['lr_embed_color']})
            
            # 创建用于当前帧映射的Adam优化器
            self.cur_map_optimizer = optim.Adam(params_cur_mapping, betas=(0.9, 0.99))
        
    
    def save_mesh(self, i, voxel_size=0.05):
        '''
        保存当前帧的三维网格模型
        Params:
            i (int): 当前帧的索引
            voxel_size (float): 网格体素的大小，控制网格的分辨率，默认为0.05
        '''
        # 确定网格模型的保存路径，路径格式为 "output/exp_name/mesh_track{i}.ply"
        mesh_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'mesh_track{}.ply'.format(i))
        
        # 根据配置文件决定使用哪种颜色渲染方法
        if self.config['mesh']['render_color']:
            color_func = self.model.render_surface_color  # 渲染表面颜色
        else:
            color_func = self.model.query_color  # 查询颜色
        
        # 调用提取网格的函数，并保存为PLY格式文件
        extract_mesh(self.model.query_sdf,          # SDF查询函数，获取距离场数据
                    self.config,                   # 配置文件
                    self.bounding_box,             # 场景的边界框
                    color_func=color_func,         # 颜色渲染函数
                    marching_cube_bound=self.marching_cube_bound,  # Marching Cubes算法的边界
                    voxel_size=voxel_size,         # 网格体素大小
                    mesh_savepath=mesh_savepath)   # 保存路径

        
    def run(self):
        '''
        主函数，负责运行整个 SLAM 系统，包括地图构建和位姿估计
        '''
        # ********************* 创建map和BA的优化器 *********************
        # 创建 Adam 优化器，用于优化编码器/解码器网络
        # 优化位姿的优化器在 tracking_render() 函数中实现
        self.create_optimizer()

        # ********************* 加载数据 *********************
        data_loader = DataLoader(self.dataset, num_workers=self.config['data']['num_workers']) # 使用 PyTorch 的 DataLoader 加载数据集

        #!  ---------------Sec 2 and Sec 3. Start Co-SLAM(tracking + Mapping) -----------------
        for i, batch in tqdm(enumerate(data_loader)):
            # Visualisation
            # 可视化rgb和深度图
            if self.config['mesh']['visualisation']:
                rgb = cv2.cvtColor(batch["rgb"].squeeze().cpu().numpy(), cv2.COLOR_BGR2RGB)
                raw_depth = batch["depth"]
                mask = (raw_depth >= self.config["cam"]["depth_trunc"]).squeeze(0)
                depth_colormap = colormap_image(batch["depth"])
                depth_colormap[:, mask] = 255.
                depth_colormap = depth_colormap.permute(1, 2, 0).cpu().numpy()
                image = np.hstack((rgb, depth_colormap))
                cv2.namedWindow('RGB-D'.format(i), cv2.WINDOW_AUTOSIZE)
                cv2.imshow('RGB-D'.format(i), image)
                key = cv2.waitKey(1)

            #? ********************* 建立初始的 地图和位姿估计 *********************
            #! -------------------- Sec 3.0 我们新增一个3.0，作为First frame mapping模块的代号，方便理解执行顺序 --------------------
            if i == 0:
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
            


            # ********************* 建立每一帧的地图和位姿估计 *********************
            # Tracking + Mapping
            else:

                #!  --------------------Sec 2. Tracking -------------------- 
                if self.config['tracking']['iter_point'] > 0:
                    # 本论文没用该方法(通过点云损失来跟踪当前帧的相机位姿)
                    self.tracking_pc(batch, i)
                # 使用当前帧的rgb损失，深度损失，sdf损失，fs损失来跟踪当前帧的相机位姿
                self.tracking_render(batch, i)
    

                #!  --------------------Sec 3. Mapping -------------------- 
                if i%self.config['mapping']['map_every']==0:  # 每5帧建一次图  ['mapping']['map_every']=5
                    self.current_frame_mapping(batch, i)
                    #! --------------------Sec 3.3 BA -------------------- 
                    self.global_BA(batch, i)

                    #! --------------------Sec 2.3 Tracked frame ----> Sec 3.1 Keyframe database --------------------
                    # 从Sec 2.3到3.1，这是一个动态过程，从tracking部分传递过来的batch，在符合keyframe_every的判断之后就可以传入，进行Pixel sampling并记录
                # Add keyframe
                if i % self.config['mapping']['keyframe_every'] == 0: # 每5帧增加一个关键帧  ['mapping']['keyframe_every']=5
                    self.keyframeDatabase.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
                    print('add keyframe:',i)
            
                #?  -------------------- Evaluation -------------------- 
                if i % self.config['mesh']['vis']==0:
                    self.save_mesh(i, voxel_size=self.config['mesh']['voxel_eval']) # 保存当前帧的网格
                    pose_relative = self.convert_relative_pose() # 将位姿转换为相对位姿
                    pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i) # 评估当前位姿
                    # 评估相对位姿
                    pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

                    if cfg['mesh']['visualisation']:
                        cv2.namedWindow('Traj:'.format(i), cv2.WINDOW_AUTOSIZE)
                        traj_image = cv2.imread(os.path.join(self.config['data']['output'], self.config['data']['exp_name'], "pose_r_{}.png".format(i)))
                        # best_traj_image = cv2.imread(os.path.join(best_logdir_scene, "pose_r_{}.png".format(i)))
                        # image_show = np.hstack((traj_image, best_traj_image))
                        image_show = traj_image
                        cv2.imshow('Traj:'.format(i), image_show)
                        key = cv2.waitKey(1)


        # 保存模型检查点
        model_savepath = os.path.join(self.config['data']['output'], self.config['data']['exp_name'], 'checkpoint{}.pt'.format(i)) 
        self.save_ckpt(model_savepath)
        # 保存最终的网格
        self.save_mesh(i, voxel_size=self.config['mesh']['voxel_final'])
        
        # 再次评估位姿
        pose_relative = self.convert_relative_pose()
        pose_evaluation(self.pose_gt, self.est_c2w_data, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i)
        pose_evaluation(self.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), i, img='pose_r', name='output_relative.txt')

        #TODO: 重建评估

# 主函数
if __name__ == '__main__':

    # ********************* 加载参数 *********************
    print('Start running...')
    # 创建一个命令行参数解析器，用于接收外部传入的参数
    parser = argparse.ArgumentParser(
        description='Arguments for running the NICE-SLAM/iMAP*.'
    )
    # 添加命令行参数 --config，用于指定配置文件路径
    parser.add_argument('--config', type=str, help='Path to config file.')
    # 添加命令行参数 --input_folder，用于指定输入文件夹路径
    parser.add_argument('--input_folder', type=str,
                        help='input folder, this have higher priority, can overwrite the one in config file')
    # 添加命令行参数 --output，用于指定输出文件夹路径
    parser.add_argument('--output', type=str,
                        help='output folder, this have higher priority, can overwrite the one in config file')
    # 解析命令行参数
    args = parser.parse_args()
    # 加载配置文件
    cfg = config.load_config(args.config)
    # 如果命令行参数指定了输出路径，则覆盖配置文件中的输出路径
    if args.output is not None:
        cfg['data']['output'] = args.output

    # ********************* 保存配置和脚本 *********************
    print("Saving config and script...")
    # 定义保存路径，保存路径包括输出目录、实验名称等信息
    save_path = os.path.join(cfg["data"]["output"], cfg['data']['exp_name'])  # Example   save_path: "output/TUM/fr_desk/demo"
    # 如果保存路径不存在，则创建该路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # 复制当前脚本到保存路径，以保留运行时的代码版本
    shutil.copy("coslam.py", os.path.join(save_path, 'coslam.py'))

    # 保存当前使用的配置文件到保存路径，保存为 JSON 格式
    with open(os.path.join(save_path, 'config.json'),"w", encoding='utf-8') as f:
        f.write(json.dumps(cfg, indent=4))

    # ************************************ 开始SLAM ************************************
    #  -------------------- Sec 1. Scene representation: 网络构建  -------------------- 
    slam = CoSLAM(cfg)  # 创建 CoSLAM 对象并传入加载的配置
    #  -------------------- Sec 2 and Sec 3. Start Co-SLAM(tracking + Mapping) -------------------- 
    slam.run()          # 调用 SLAM 系统的 run() 方法，启动整个跟踪和映射过程
