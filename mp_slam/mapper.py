# Use dataset object

import torch
import time
import os
import random

class Mapper():
    """
    Mapper类用于管理SLAM（同步定位与建图）中的映射过程。
    它负责映射新观测到的帧，优化相机姿态，并执行全局束调整。

    属性:
        config (dict): 映射过程的配置参数。
        slam (object): SLAM系统的实例，包含模型、跟踪和映射信息。
        model (nn.Module): 用于SLAM的神经网络模型。
        tracking_idx (torch.Tensor): 当前跟踪帧的索引。
        mapping_idx (torch.Tensor): 当前映射帧的索引。
        mapping_first_frame (torch.Tensor): 首帧映射标志。
        keyframe (KeyFrameDatabase): 关键帧数据库，用于管理关键帧。
        map_optimizer (torch.optim.Optimizer): 用于映射优化的优化器。
        device (torch.device): 设备信息（CPU或GPU）。
        dataset (Dataset): 用于映射的输入数据集。
        est_c2w_data (list): 当前估计的相机到世界坐标的转换矩阵列表。
        est_c2w_data_rel (list): 当前估计的相对相机位姿矩阵列表。
    """
    def __init__(self, config, SLAM) -> None:
        self.config = config                    # 保存配置文件
        self.slam = SLAM                        # 保存SLAM对象的引用
        self.model = SLAM.model                 # 保存SLAM模型
        self.tracking_idx = SLAM.tracking_idx   # 当前的跟踪帧索引
        self.mapping_idx = SLAM.mapping_idx     # 当前的映射帧索引
        self.mapping_first_frame = SLAM.mapping_first_frame   # 标记是否已经完成首帧映射
        self.keyframe = SLAM.keyframeDatabase   # 保存关键帧数据库的引用
        self.map_optimizer = SLAM.map_optimizer # 保存映射优化器
        self.device = SLAM.device               # 保存设备信息
        self.dataset = SLAM.dataset             # 保存输入数据集
        self.est_c2w_data = SLAM.est_c2w_data   # 当前估计的相机到世界坐标的转换矩阵列表
        self.est_c2w_data_rel = SLAM.est_c2w_data_rel   # 当前估计的相对相机位姿矩阵列表

    def first_frame_mapping(self, batch, n_iters=100):
        """
        执行首帧映射。

        参数:
            batch (dict): 包含首帧数据的字典，包括c2w、rgb、depth、direction等。
            n_iters (int): 首帧映射的迭代次数，默认是100。

        返回:
            ret (dict): 映射后的结果字典。
            loss (float): 映射过程中的损失值。
        """
        # 打印日志信息，说明正在进行首帧映射
        print('First frame mapping...')
        # 检查当前帧是否为首帧
        if batch['frame_id'] != 0:
            raise ValueError('First frame mapping must be the first frame!')
        # 获取并设置首帧的相机到世界坐标系的变换矩阵 (c2w)
        c2w = batch['c2w'].to(self.device)
        self.est_c2w_data[0] = c2w
        self.est_c2w_data_rel[0] = c2w

        # 设置模型为训练模式
        self.model.train()

        # 进行训练迭代
        for i in range(n_iters):
            self.map_optimizer.zero_grad()  # 清零优化器的梯度
            # 从SLAM系统中选择样本
            indice = self.slam.select_samples(self.slam.dataset.H, self.slam.dataset.W, self.config['mapping']['sample'])
            indice_h, indice_w = indice % (self.slam.dataset.H), indice // (self.slam.dataset.H)
            # 获取样本的方向、RGB和深度数据，并将其移动到设备上
            rays_d_cam = batch['direction'][indice_h, indice_w, :].to(self.device)
            target_s = batch['rgb'][indice_h, indice_w, :].to(self.device)
            target_d = batch['depth'][indice_h, indice_w].to(self.device).unsqueeze(-1)

            # 计算光线的原点和方向
            rays_o = c2w[None, :3, -1].repeat(self.config['mapping']['sample'], 1)
            rays_d = torch.sum(rays_d_cam[..., None, :] * c2w[:3, :3], -1)

            # Forward
            ret = self.model.forward(rays_o.to(self.device), rays_d.to(self.device), target_s, target_d)
            # 计算损失并进行反向传播
            loss = self.slam.get_loss_from_ret(ret)
            loss.backward()
            self.map_optimizer.step()
        
        # First frame will always be a keyframe
        self.keyframe.add_keyframe(batch, filter_depth=self.config['mapping']['filter_depth'])
        # if self.config['mapping']['first_mesh']:
        #     self.slam.save_mesh(0)
        
        print('First frame mapping done')
        # 更新首帧的映射状态
        self.mapping_first_frame[0] = 1
        return ret, loss

    def global_BA(self, batch, cur_frame_id):
        '''
        包括所有关键帧和当前帧的全局束调整
        Params:
            batch['c2w']: ground truth camera pose [1, 4, 4]
            batch['rgb']: rgb image [1, H, W, 3]
            batch['depth']: depth image [1, H, W, 1]
            batch['direction']: view direction [1, H, W, 3]
            cur_frame_id: current frame id
        '''
        pose_optimizer = None

        # all the KF poses: 0, 5, 10, ...
        poses = torch.stack([self.est_c2w_data[i] for i in range(0, cur_frame_id, self.config['mapping']['keyframe_every'])])
        
        # 所有 KF 的帧 ID，用于优化后的更新姿势
        frame_ids_all = torch.tensor(list(range(0, cur_frame_id, self.config['mapping']['keyframe_every'])))

        if len(self.keyframe.frame_ids) < 2:
            # 如果关键帧少于两个，固定所有关键帧的姿态
            poses_fixed = torch.nn.parameter.Parameter(poses).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]
            poses_all = torch.cat([poses_fixed, current_pose], dim=0)
        
        else:
            # 否则进行优化
            poses_fixed = torch.nn.parameter.Parameter(poses[:1]).to(self.device)
            current_pose = self.est_c2w_data[cur_frame_id][None,...]

            if self.config['mapping']['optim_cur']:
                cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(torch.cat([poses[1:], current_pose]))
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim], dim=0)

            else:
                cur_rot, cur_trans, pose_optimizer, = self.slam.get_pose_param_optim(poses[1:])
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans).to(self.device)
                poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)
        
        # 设置优化
        self.map_optimizer.zero_grad()
        if pose_optimizer is not None:
            pose_optimizer.zero_grad()
        
        # 将当前帧的光线数据进行预处理
        current_rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        current_rays = current_rays.reshape(-1, current_rays.shape[-1])

        

        for i in range(self.config['mapping']['iters']):

            # 使用真实帧 ID 对光线进行采样
            # rays [bs, 7]
            # frame_ids [bs]
            rays, ids = self.keyframe.sample_global_rays(self.config['mapping']['sample'])

            #TODO: Checkpoint...
            # 随机选择当前帧的一些像素进行训练
            idx_cur = random.sample(range(0, self.slam.dataset.H * self.slam.dataset.W),max(self.config['mapping']['sample'] // len(self.keyframe.frame_ids), self.config['mapping']['min_pixels_cur']))
            current_rays_batch = current_rays[idx_cur, :]

            # 合并关键帧光线和当前帧光线
            rays = torch.cat([rays, current_rays_batch], dim=0) # N, 7
            ids_all = torch.cat([ids//self.config['mapping']['keyframe_every'], -torch.ones((len(idx_cur)))]).to(torch.int64)


            rays_d_cam = rays[..., :3].to(self.device)
            target_s = rays[..., 3:6].to(self.device)
            target_d = rays[..., 6:7].to(self.device)

            # [N, Bs, 1, 3] * [N, 1, 3, 3] = (N, Bs, 3)
            # 计算光线在各个姿态下的方向
            rays_d = torch.sum(rays_d_cam[..., None, None, :] * poses_all[ids_all, None, :3, :3], -1)
            rays_o = poses_all[ids_all, None, :3, -1].repeat(1, rays_d.shape[1], 1).reshape(-1, 3)
            rays_d = rays_d.reshape(-1, 3)

            # 前向传播
            ret = self.model.forward(rays_o, rays_d, target_s, target_d)

            # 计算损失并进行反向传播
            loss = self.slam.get_loss_from_ret(ret, smooth=True)
            
            loss.backward(retain_graph=True)
            
            # 每隔一定步骤更新映射优化器
            if (i + 1) % self.config["mapping"]["map_accum_step"] == 0:
               
                if (i + 1) > self.config["mapping"]["map_wait_step"]:
                    self.map_optimizer.step()
                else:
                    print('Wait update')
                self.map_optimizer.zero_grad()

            # 每隔一定步骤更新姿态优化器
            if pose_optimizer is not None and (i + 1) % self.config["mapping"]["pose_accum_step"] == 0:
                pose_optimizer.step()
                # 获取 SE3 姿态进行前向传播
                pose_optim = self.slam.matrix_from_tensor(cur_rot, cur_trans)
                pose_optim = pose_optim.to(self.device)
                # 所以当前姿势总是不变的
                if self.config['mapping']['optim_cur']: # 如果优化当前姿态，更新姿态列表
                    poses_all = torch.cat([poses_fixed, pose_optim], dim=0)
                
                else:
                    # 否则将当前姿态加入到姿态列表中
                    current_pose = self.est_c2w_data[cur_frame_id][None,...]
                    # SE3 poses

                    poses_all = torch.cat([poses_fixed, pose_optim, current_pose], dim=0)


                # 清零姿态优化器的梯度
                pose_optimizer.zero_grad()
        
        # 更新优化后的姿态
        if pose_optimizer is not None and len(frame_ids_all) > 1:
            for i in range(len(frame_ids_all[1:])):
                self.est_c2w_data[int(frame_ids_all[i+1].item())] = self.slam.matrix_from_tensor(cur_rot[i:i+1], cur_trans[i:i+1]).detach().clone()[0]
        
            if self.config['mapping']['optim_cur']:
                print('Update current pose')
                self.est_c2w_data[cur_frame_id] = self.slam.matrix_from_tensor(cur_rot[-1:], cur_trans[-1:]).detach().clone()[0]
    
    def convert_relative_pose(self, idx):
        """
        将相对位姿转换为绝对位姿。
        
        参数:
            idx (int): 当前处理的帧索引，所有帧的绝对位姿都将被计算到这个索引之前。
        
        返回:
            dict: 包含所有帧的绝对位姿的字典。键为帧索引，值为相应的绝对位姿矩阵。
        """
        poses = {}
        
        # 遍历所有在 idx 之前的帧
        for i in range(len(self.est_c2w_data[:idx])):
            # 检查当前帧是否为关键帧
            if i % self.config['mapping']['keyframe_every'] == 0:
                # 关键帧的绝对位姿直接使用
                poses[i] = self.est_c2w_data[i]
            else:
                # 计算当前帧的绝对位姿
                # 确定当前帧的最近的关键帧的 ID
                kf_id = i // self.config['mapping']['keyframe_every']
                kf_frame_id = kf_id * self.config['mapping']['keyframe_every']
                
                # 获取对应关键帧的绝对位姿
                c2w_key = self.est_c2w_data[kf_frame_id]
                
                # 获取当前帧相对于其最近关键帧的相对位姿
                delta = self.est_c2w_data_rel[i]
                
                # 通过矩阵乘法计算当前帧的绝对位姿
                poses[i] = delta @ c2w_key  # delta 和 c2w_key 矩阵相乘
        
        return poses

    def run(self):

        # Start mapping
        while self.tracking_idx[0]< len(self.dataset)-1:
            if self.tracking_idx[0] == 0 and self.mapping_first_frame[0] == 0:
                batch = self.dataset[0]
                self.first_frame_mapping(batch, self.config['mapping']['first_iters'])
                time.sleep(0.1)
            else:
                while self.tracking_idx[0] <= self.mapping_idx[0] + self.config['mapping']['map_every']:
                    time.sleep(0.4)
                current_map_id = int(self.mapping_idx[0] + self.config['mapping']['map_every'])
                batch = self.dataset[current_map_id]
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v[None, ...]
                    else:
                        batch[k] = torch.tensor([v])
                self.global_BA(batch, current_map_id)
                self.mapping_idx[0] = current_map_id
            
                if self.mapping_idx[0] % self.config['mapping']['keyframe_every'] == 0:
                    self.keyframe.add_keyframe(batch)
            
                if self.mapping_idx[0] % self.config['mesh']['vis']==0:
                    idx = int(self.mapping_idx[0])
                    self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_eval'])
                    pose_relative = self.convert_relative_pose(idx)
                    self.slam.pose_eval_func()(self.slam.pose_gt, self.est_c2w_data[:idx], 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx)
                    self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='pose_r', name='output_relative.txt')
                
                time.sleep(0.2)

        idx = int(self.tracking_idx[0])       
        self.slam.save_mesh(idx, voxel_size=self.config['mesh']['voxel_final'])
        pose_relative = self.convert_relative_pose(idx)
        self.slam.pose_eval_func()(self.slam.pose_gt, self.est_c2w_data[:idx], 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx)
        self.slam.pose_eval_func()(self.slam.pose_gt, pose_relative, 1, os.path.join(self.config['data']['output'], self.config['data']['exp_name']), idx, img='pose_r', name='output_relative.txt')

        
        
        
