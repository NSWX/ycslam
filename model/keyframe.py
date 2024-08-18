import torch
import numpy as np
import random


class KeyFrameDatabase(object):
    """
    KeyFrameDatabase 类用于在视觉SLAM（同时定位与地图构建）或其他基于多视角图像的任务中，
        处理和管理关键帧数据。此类提供了关键帧的存储、采样、选择和查询等功能。

    Attributes:
        - config (dict): 包含相机配置和其他参数的配置字典。
        - H (int): 图像的高度。
        - W (int): 图像的宽度。
        - num_kf (int): 要保存的关键帧数量。
        - num_rays_to_save (int): 每个关键帧保存的光线数量。
        - device (torch.device): PyTorch 中使用的设备（例如 'cpu' 或 'cuda'）。
        - keyframes (dict): 存储关键帧信息的字典。
        - rays (torch.Tensor): 用于存储采样的光线数据的张量。
        - frame_ids (torch.Tensor): 存储帧ID的张量。
    """

    def __init__(self, config, H, W, num_kf, num_rays_to_save, device) -> None:
        """
        初始化 KeyFrameDatabase 类。

        参数:
            - config (dict): 包含相机配置和其他参数的配置字典。
            - H (int): 图像的高度。
            - W (int): 图像的宽度。
            - num_kf (int): 要保存的关键帧数量。
            - num_rays_to_save (int): 每个关键帧保存的光线数量。
            - device (torch.device): PyTorch 中使用的设备（例如 'cpu' 或 'cuda'）。
        """
        # 初始化关键帧数据库的相关参数
        self.config = config  # 相机配置文件等参数
        self.keyframes = {}  # 用于存储关键帧的字典
        self.device = device  # 计算设备（如 'cuda' 或 'cpu'）
        # 存储所有关键帧的光线信息，形状为 (num_kf, num_rays_to_save, 7)
        self.rays = torch.zeros((num_kf, num_rays_to_save, 7))
        self.num_rays_to_save = num_rays_to_save  # 每个关键帧要保存的光线数量
        self.frame_ids = None  # 存储关键帧的帧 ID
        self.H = H  # 图像高度
        self.W = W  # 图像宽度

    def __len__(self):
        """
        返回当前存储的关键帧数量。

        返回:
            - int: 存储的关键帧数量。
        """
        return len(self.frame_ids)

    def get_length(self):
        """
        返回当前存储的关键帧数量。
        这是 `__len__` 方法的一个包装器。

        返回:
            - int: 存储的关键帧数量。
        """
        return self.__len__()

    def sample_single_keyframe_rays(self, rays, option='random'):
        """
        从给定的光线数据中采样光线。

        参数:
            - rays (torch.Tensor): 包含光线数据的张量，形状为 (H*W, 7)。
            - option (str): 采样策略，可以是 'random' 或 'filter_depth'。

        返回:
            - torch.Tensor: 采样后的光线数据。
        """
        if option == 'random':
            # 随机采样
            idxs = random.sample(range(0, self.H * self.W), self.num_rays_to_save)
        elif option == 'filter_depth':
            # 基于深度信息进行采样
            valid_depth_mask = (rays[..., -1] > 0.0) & (rays[..., -1] <= self.config["cam"]["depth_trunc"])
            rays_valid = rays[valid_depth_mask, :]  # 获取深度有效的光线
            num_valid = len(rays_valid)
            idxs = random.sample(range(0, num_valid), self.num_rays_to_save)

        else:
            raise NotImplementedError()
        # 根据采样的索引返回光线
        rays = rays[:, idxs]
        return rays

    def attach_ids(self, frame_ids):
        """
        将帧 ID 附加到当前的帧 ID 列表中。

        参数:
            - frame_ids (torch.Tensor): 包含要附加的帧 ID 的张量。
        """
        if self.frame_ids is None:
            self.frame_ids = frame_ids
        else:
            # 将新帧 ID 附加到已有的帧 ID 列表中
            self.frame_ids = torch.cat([self.frame_ids, frame_ids], dim=0)

    def add_keyframe(self, batch, filter_depth=False):
        """
        将关键帧光线添加到关键帧数据库中。

        参数:
            - batch (dict): 包含当前帧数据的批次，包括 'direction'、'rgb' 和 'depth'。
            - filter_depth (bool): 是否根据深度过滤光线。
        """
        # 获取 batch 中的方向、颜色和深度信息，并组合成光线数据
        rays = torch.cat([batch['direction'], batch['rgb'], batch['depth'][..., None]], dim=-1)
        rays = rays.reshape(1, -1, rays.shape[-1])

        # 根据是否过滤深度信息进行采样
        if filter_depth:
            rays = self.sample_single_keyframe_rays(rays, 'filter_depth')
        else:
            rays = self.sample_single_keyframe_rays(rays)

        # 如果 frame_id 不是 Tensor 类型，则转换为 Tensor
        if not isinstance(batch['frame_id'], torch.Tensor):
            batch['frame_id'] = torch.tensor([batch['frame_id']])

        # 将当前帧 ID 添加到 frame_ids 列表中
        self.attach_ids(batch['frame_id'])

        # 存储采样的光线信息
        self.rays[len(self.frame_ids) - 1] = rays

    def sample_global_rays(self, bs):
        """
        从所有关键帧中采样光线。

        参数:
            - bs (int): 要采样的光线数量。

        返回:
            - sample_rays (torch.Tensor): 采样的光线数据。
            - frame_ids (torch.Tensor): 对应的帧 ID。
        """
        num_kf = self.__len__()  # 获取关键帧数量
        # 从所有关键帧的光线中随机选择 bs 个光线
        idxs = torch.tensor(random.sample(range(num_kf * self.num_rays_to_save), bs))
        # 根据索引获取光线数据
        sample_rays = self.rays[:num_kf].reshape(-1, 7)[idxs]

        # 获取对应的帧 ID
        frame_ids = self.frame_ids[idxs // self.num_rays_to_save]

        return sample_rays, frame_ids

    def sample_global_keyframe(self, window_size, n_fixed=1):
        """
        全局采样关键帧。

        参数:
            - window_size (int): 采样窗口大小。
            - n_fixed (int): 固定采样的最后几个关键帧数量。

        返回:
            - select_rays (torch.Tensor): 选择的光线数据。
            - frame_ids (torch.Tensor): 选择的帧 ID。
        """
        if window_size >= len(self.frame_ids):
            # 如果窗口大小大于等于帧数量，返回所有光线和帧 ID
            return self.rays[:len(self.frame_ids)], self.frame_ids

        current_num_kf = len(self.frame_ids)  # 当前关键帧数量
        last_frame_ids = self.frame_ids[-n_fixed:]  # 最后 n_fixed 个帧 ID

        # 随机采样窗口大小范围内的帧 ID
        idx = random.sample(range(0, len(self.frame_ids) - n_fixed), window_size)

        # 将最后 n_fixed 个关键帧包括在内
        idx_rays = idx + list(range(current_num_kf - n_fixed, current_num_kf))
        select_rays = self.rays[idx_rays]

        return select_rays, torch.cat([self.frame_ids[idx], last_frame_ids], dim=0)

    @torch.no_grad()
    def sample_overlap_keyframe(self, batch, frame_id, est_c2w_list, k_frame, n_samples=16, n_pixel=100, dataset=None):
        """
        NICE-SLAM 策略，用于从所有先前帧中选择重叠的关键帧。

        参数:
            - batch (dict): 当前帧的信息。
            - frame_id (int): 当前帧的 ID。
            - est_c2w_list (list): 所有帧的估计 c2w 矩阵列表。
            - k_frame (int): 用于 BA 的关键帧数量（窗口大小）。
            - n_samples (int): 每条光线的采样点数。
            - n_pixel (int): 计算重叠的像素数量。
            - dataset: 数据集对象。

        返回:
            - torch.Tensor: 选择的光线数据。
            - list: 选择的关键帧 ID 列表。
        """
        c2w_est = est_c2w_list[frame_id]  # 获取当前帧的估计 c2w 矩阵

        # 随机从图像中选取 n_pixel 个像素的索引
        indices = torch.randint(dataset.H * dataset.W, (n_pixel,))
        # 获取对应像素的方向和深度信息
        rays_d_cam = batch['direction'].reshape(-1, 3)[indices].to(self.device)
        target_d = batch['depth'].reshape(-1, 1)[indices].repeat(1, n_samples).to(self.device)

        # 计算光线方向和起点
        rays_d = torch.sum(rays_d_cam[..., None, :] * c2w_est[:3, :3], -1)
        rays_o = c2w_est[None, :3, -1].repeat(rays_d.shape[0], 1).to(self.device)

        # 计算光线在深度范围内的采样点
        t_vals = torch.linspace(0., 1., steps=n_samples).to(target_d)
        near = target_d * 0.8
        far = target_d + 0.5
        z_vals = near * (1. - t_vals) + far * (t_vals)
        pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]  # [N_rays, N_samples, 3]
        pts_flat = pts.reshape(-1, 3).cpu().numpy()  # 展平为 2D 数组

        key_frame_list = []

        for i, frame_id in enumerate(self.frame_ids):
            frame_id = int(frame_id.item())
            c2w = est_c2w_list[frame_id].cpu().numpy()  # 获取关键帧的 c2w 矩阵
            w2c = np.linalg.inv(c2w)  # 计算相机坐标系到世界坐标系的转换矩阵
            ones = np.ones_like(pts_flat[:, 0]).reshape(-1, 1)
            pts_flat_homo = np.concatenate([pts_flat, ones], axis=1).reshape(-1, 4, 1)  # 将点转换为齐次坐标
            cam_cord_homo = w2c @ pts_flat_homo  # 将点从世界坐标转换到相机坐标
            cam_cord = cam_cord_homo[:, :3]  # 获取相机坐标

            # 相机内参矩阵
            K = np.array([[self.config['cam']['fx'], .0, self.config['cam']['cx']],
                          [.0, self.config['cam']['fy'], self.config['cam']['cy']],
                          [.0, .0, 1.0]]).reshape(3, 3)
            cam_cord[:, 0] *= -1  # 修正 x 方向
            uv = K @ cam_cord  # 投影到图像平面
            z = uv[:, -1:] + 1e-5  # 获取深度值并防止除零
            uv = uv[:, :2] / z  # 归一化图像坐标
            uv = uv.astype(np.float32)

            edge = 20  # 边缘距离阈值
            # 检查投影点是否在图像边界内
            mask = (uv[:, 0] < self.config['cam']['W'] - edge) * (uv[:, 0] > edge) * \
                   (uv[:, 1] < self.config['cam']['H'] - edge) * (uv[:, 1] > edge)
            mask = mask & (z[:, :, 0] < 0)
            mask = mask.reshape(-1)

            percent_inside = mask.sum() / uv.shape[0]  # 计算在图像内的点的比例
            key_frame_list.append(
                {'id': frame_id, 'percent_inside': percent_inside, 'sample_id': i})

        # 按照在图像内的点的比例降序排列关键帧列表
        key_frame_list = sorted(
            key_frame_list, key=lambda i: i['percent_inside'], reverse=True)
        # 选择符合条件的关键帧，并随机选择 k_frame 个关键帧
        selected_keyframe_list = [dic['sample_id']
                                  for dic in key_frame_list if dic['percent_inside'] > 0.00]
        selected_keyframe_list = list(np.random.permutation(
            np.array(selected_keyframe_list))[:k_frame])

        last_id = len(self.frame_ids) - 1  # 获取最后一个关键帧的索引

        # 确保最后一个关键帧被包含在选择列表中
        if last_id not in selected_keyframe_list:
            selected_keyframe_list.append(last_id)

        return self.rays[selected_keyframe_list], selected_keyframe_list  # 返回选择的光线和关键帧列表
