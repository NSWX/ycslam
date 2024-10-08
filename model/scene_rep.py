# package imports
import torch
import torch.nn as nn

# Local imports
from .encodings import get_encoder
from .decoder import ColorSDFNet, ColorSDFNet_v2
from .utils import sample_pdf, batchify, get_sdf_loss, mse2psnr, compute_loss

class JointEncoding(nn.Module):
    def __init__(self, config, bound_box):
        super(JointEncoding, self).__init__()   # 调用父类的初始化方法，确保父类的属性和方法被正确继承。
        self.config = config                    # 保存传入的配置文件 `config`，该配置文件包含了网络结构、参数设置等信息。
        self.bounding_box = bound_box           # 保存传入的边界框 `bound_box`，定义了场景中需要处理的空间范围。
        self.get_resolution()                   # 调用 `get_resolution` 方法，从配置文件中获取体素的深度和颜色分辨率，用于场景表示。
        
        # ********************* 1.1 编码 *********************
        """
        首先,调用 `get_encoding` 方法，从配置文件中获取编码方案: 1. parametric encoding用HashGrid 2. coordinate encoding用OneBlob
        然后,通过tiny-cuda-nn库实现这些编码网络，用于将空间坐标或特征映射到高维空间，以捕捉更丰富的场景信息。
        参考https://github.com/NVlabs/tiny-cuda-nn/blob/master/src/encoding.cu
        """
        self.get_encoding(config)

        # ********************* 1.2 解码 *********************
        """
        首先,从config中获得解码网络的各个参数
        然后,为颜色color和深度sdf各创建一个2层的MLP网络,激活函数都是ReLU
        sdf:   最后输出维度为16, 即预测的SDF值(1维) + 特征向量h值(15维)
        color: 最后输出维度为3, 即预测的RGB值
        """
        self.get_decoder(config)
        

    def get_resolution(self):
        '''
        根据场景的边界框和配置文件中的参数，计算用于3D场景表示的体素（Voxel）分辨率。
        这些分辨率分别用于深度（SDF）和颜色信息的编码和解码。具体来说，它确定了网格在空间中的分辨率，即每个体素的大小。
        '''
        dim_max = (self.bounding_box[:,1] - self.bounding_box[:,0]).max()   # 计算边界框中最长的维度（X、Y、Z中的最大值）。这是用于确定网格分辨率的参考长度。
        
        if self.config['grid']['voxel_sdf'] > 10:                                    #! grid 作为一个键，包含了与网格相关的所有参数
            self.resolution_sdf = self.config['grid']['voxel_sdf']                   # 如果 grid 下的具体参数 voxel_sdf 大于10，直接将其作为SDF的分辨率。
        else:
            self.resolution_sdf = int(dim_max / self.config['grid']['voxel_sdf'])    # 否则，将最长维度 `dim_max` 除以配置中的 `voxel_sdf` 参数，得到实际的分辨率（体素数目）。
        
        if self.config['grid']['voxel_color'] > 10:
            self.resolution_color = self.config['grid']['voxel_color']
        else:
            self.resolution_color = int(dim_max / self.config['grid']['voxel_color'])
        
        print('SDF resolution:', self.resolution_sdf)

    # 对应 3.1 编码
    def get_encoding(self, config):
        '''
        获取场景表示的编码

        以tum.yaml为例
        pos enc: 'OneBlob'
        grid enc: 'HashGrid'
        grid oneGrid: 'True'
        '''
        # 坐标编码
        self.embedpos_fn, self.input_ch_pos = get_encoder(config['pos']['enc'], n_bins=self.config['pos']['n_bins'])
        # 获取坐标编码器。这里 `config['pos']['enc']` 定义了坐标编码的方法，例如'OneBlob'。
        # `n_bins` 指定了编码的分辨率（或分块数）。`get_encoder` 返回编码函数和输入通道数（input_ch_pos）。

        # 稀疏参数编码 （SDF）
        self.embed_fn, self.input_ch = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_sdf)
        # 获取用于SDF（Signed Distance Function）编码的稀疏参数编码器。这里 `config['grid']['enc']` 定义了编码方法，例如'HashGrid'。
        # `log2_hashmap_size` 指定哈希表的大小（以2为底的对数形式），`desired_resolution` 指定编码器的分辨率（使用之前计算的SDF分辨率）。

        # 稀疏参数编码（颜色）
        if not self.config['grid']['oneGrid']:  # 如果配置文件中 `grid` 的 `oneGrid` 参数为 `False`，即SDF和颜色不使用同一个网格编码器，则需要为颜色创建一个单独的稀疏参数编码器。
            print('Color resolution:', self.resolution_color)   # 打印颜色分辨率 `self.resolution_color`，然后调用 `get_encoder` 来获取颜色编码器 `embed_fn_color` 和对应的输入通道数 `input_ch_color`。
            self.embed_fn_color, self.input_ch_color = get_encoder(config['grid']['enc'], log2_hashmap_size=config['grid']['hash_size'], desired_resolution=self.resolution_color)

    #! 对应 3.2 解码
    def get_decoder(self, config):
        '''
        获取场景表示的解码器

        以tum.yaml为例
        grid oneGrid: True
        '''

        # 处理嵌入的空间位置和几何特征，以生成颜色和SDF值
        # ********************* 根据论文公式(2)(3)，处理嵌入的空间位置和几何特征，以生成颜色和SDF值*********************
        if not self.config['grid']['oneGrid']:
            self.decoder = ColorSDFNet(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        else:
            self.decoder = ColorSDFNet_v2(config, input_ch=self.input_ch, input_ch_pos=self.input_ch_pos)
        
        self.color_net = batchify(self.decoder.color_net, None)
        self.sdf_net = batchify(self.decoder.sdf_net, None)

    # 对应论文，将sdf值转成weights权重
    def sdf2weights(self, sdf, z_vals, args=None):
        '''
        Convert signed distance function to weights.

        Params:
            sdf: [N_rays, N_samples]
            z_vals: [N_rays, N_samples]
        Returns:
            weights: [N_rays, N_samples]
        '''
        # ********************* 根据论文公式(5)用两个sigmoid计算权重 *********************
        weights = torch.sigmoid(sdf / args['training']['trunc']) * torch.sigmoid(-sdf / args['training']['trunc'])

        signs = sdf[:, 1:] * sdf[:, :-1]
        mask = torch.where(signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs))
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds) # The first surface
        mask = torch.where(z_vals < z_min + args['data']['sc_factor'] * args['training']['trunc'], torch.ones_like(z_vals), torch.zeros_like(z_vals))

        weights = weights * mask
        return weights / (torch.sum(weights, axis=-1, keepdims=True) + 1e-8)
    
    # 内部函数
    def raw2outputs(self, raw, z_vals, white_bkgd=False):
        '''
        使用从 sdf 计算的权重执行体积渲染。

        Params:
            raw: [N_rays, N_samples, 4]
            z_vals: [N_rays, N_samples]
        Returns:
            rgb_map: [N_rays, 3]
            disp_map: [N_rays]
            acc_map: [N_rays]
            weights: [N_rays, N_samples]
        '''
        rgb = torch.sigmoid(raw[...,:3])  # [N_rays, N_samples, 3]
        weights = self.sdf2weights(raw[..., 3], z_vals, args=self.config)	# 通过raw的第四列深度值sdf,得到论文公式5的权重  [2048,85]
        
        #! ********************* 根据论文公式(4)计算rendering颜色和深度 *********************
        rgb_map = torch.sum(weights[...,None] * rgb, -2)  # [N_rays, 3] [2048,3]
        depth_map = torch.sum(weights * z_vals, -1) # 论文4式，计算深度图    [2048]

        depth_var = torch.sum(weights * torch.square(z_vals - depth_map.unsqueeze(-1)), dim=-1) # 深度方差,表示每条射线深度的变化情况
        disp_map = 1./torch.max(1e-10 * torch.ones_like(depth_map), depth_map / torch.sum(weights, -1)) # 视差图disparity map,相机到物体的距离的倒数,表达物体的远近
        acc_map = torch.sum(weights, -1) # 累积权重图，表示沿每条射线的累积权重

        if white_bkgd:
            rgb_map = rgb_map + (1.-acc_map[...,None])

        return rgb_map, disp_map, acc_map, weights, depth_map, depth_var
    
    # 外部查询函数，在coslam.py中调用
    def query_sdf(self, query_points, return_geo=False, embed=False):
        '''
        获取查询点的 SDF 值
        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            sdf: [N_rays, N_samples]
            geo_feat: [N_rays, N_samples, channel]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])
  
        embedded = self.embed_fn(inputs_flat)
        if embed:
            return torch.reshape(embedded, list(query_points.shape[:-1]) + [embedded.shape[-1]])

        embedded_pos = self.embedpos_fn(inputs_flat)
        out = self.sdf_net(torch.cat([embedded, embedded_pos], dim=-1))
        sdf, geo_feat = out[..., :1], out[..., 1:]

        sdf = torch.reshape(sdf, list(query_points.shape[:-1]))
        if not return_geo:
            return sdf
        geo_feat = torch.reshape(geo_feat, list(query_points.shape[:-1]) + [geo_feat.shape[-1]])

        return sdf, geo_feat
    
    # 外部查询函数，在coslam.py中调用
    def query_color(self, query_points):
        return torch.sigmoid(self.query_color_sdf(query_points)[..., :3])
      
    # 外部查询函数，在coslam.py中调用
    def query_color_sdf(self, query_points):
        '''
        查询 query_points 的颜色和 sdf。

        Params:
            query_points: [N_rays, N_samples, 3]
        Returns:
            raw: [N_rays, N_samples, 4]
        '''
        inputs_flat = torch.reshape(query_points, [-1, query_points.shape[-1]])

        # ********************* encoder *********************
        embed = self.embed_fn(inputs_flat)
        embe_pos = self.embedpos_fn(inputs_flat)
        # ********************* decoder 得到颜色和深度 *********************
        if not self.config['grid']['oneGrid']:
            embed_color = self.embed_fn_color(inputs_flat)
            return self.decoder(embed, embe_pos, embed_color)
        return self.decoder(embed, embe_pos)
    
    # 内部函数
    def run_network(self, inputs):
        """
        在一批输入上运行网络。

        Params:
            inputs: [N_rays, N_samples, 3]
        Returns:
            outputs: [N_rays, N_samples, 4]
        """
        inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])	# [174080,3] <=== [2048,85,3]
        
        # Normalize the input to [0, 1] (TCNN convention)  # Normalize the input to [0, 1] (TCNN convention)归一化处理 2048根射线,每根射线85个采样点到[0,1]
        if self.config['grid']['tcnn_encoding']:
            inputs_flat = (inputs_flat - self.bounding_box[:, 0]) / (self.bounding_box[:, 1] - self.bounding_box[:, 0])
    	 # 将inputs_flat分成较小的批次，然后对每个批次使用query_color_sdf函数(也就是encoder+decoder网络)
        
         # 得到每个点的深度和颜色信息	

        outputs_flat = batchify(self.query_color_sdf, None)(inputs_flat)
        outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])

        return outputs
    
    # 外部函数，在coslam.py中调用
    def render_surface_color(self, rays_o, normal):
        '''
        渲染点的表面颜色。
        Params:
            points: [N_rays, 1, 3]
            normal: [N_rays, 3]
        '''
        n_rays = rays_o.shape[0]
        trunc = self.config['training']['trunc']
        z_vals = torch.linspace(-trunc, trunc, steps=self.config['training']['n_range_d']).to(rays_o)
        z_vals = z_vals.repeat(n_rays, 1)
        # Run rendering pipeline
        
        pts = rays_o[...,:] + normal[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        rgb, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])
        return rgb
    
    # 内部函数，神经渲染流程的实现，用于处理光线并返回渲染结果，在forward()函数中被调用
    def render_rays(self, rays_o, rays_d, target_d=None):
        '''
        Params:
            rays_o: [N_rays, 3]
            rays_d: [N_rays, 3]
            target_d: [N_rays, 1]
        '''

        # 首先拿到射线的数量
        n_rays = rays_o.shape[0]
        
        
        '''
        以tum.yaml为例
            training range_d: 0.25
            training n_range_d: 21
        '''

        #! -------------------- Sec 2.1 && 3.2 Ray samping -------------------- 
        # A. 射线深度采样，确保深度为正值
        if target_d is not None:
            # 如果有目标深度 target_d，在目标深度附近取样
            # 在深度[-range_d,range_d]范围内生成n_range_d个等间隔张量
            z_samples = torch.linspace(-self.config['training']['range_d'], self.config['training']['range_d'], steps=self.config['training']['n_range_d']).to(target_d) 
            z_samples = z_samples[None, :].repeat(n_rays, 1) + target_d
            # 将目标深度为负的那些取样点改为在深度near到far的范围内生成n_range_d个等间隔张量
            z_samples[target_d.squeeze()<=0] = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], steps=self.config['training']['n_range_d']).to(target_d) 

            if self.config['training']['n_samples_d'] > 0:
                # 符合此判断，则再在深度near到far的范围内生成n_samples_d个取样点
                z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples_d'])[None, :].repeat(n_rays, 1).to(rays_o)
                # 深度值排序
                z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            else:
                z_vals = z_samples
        else:
            z_vals = torch.linspace(self.config['cam']['near'], self.config['cam']['far'], self.config['training']['n_samples']).to(rays_o)
            z_vals = z_vals[None, :].repeat(n_rays, 1) # [n_rays, n_samples]
        # B. 如果没有目标深度，或者设置了额外的采样深度 (n_samples_d)，则在相机的视野范围内进行均匀取样

        # C. 是否要进行深度扰动
        if self.config['training']['perturb'] > 0.:
            mids = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            upper = torch.cat([mids, z_vals[...,-1:]], -1)
            lower = torch.cat([z_vals[...,:1], mids], -1)
            z_vals = lower + (upper - lower) * torch.rand(z_vals.shape).to(rays_o)

        # D. 执行神经网络渲染
        pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples, 3]
        raw = self.run_network(pts)
        # E. 将原始数据转换为具体的渲染结果，如RGB图像、深度图、不透明度累积和深度方差
        rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # Importance sampling，yaml里默认是0，不执行
        if self.config['training']['n_importance'] > 0:

            rgb_map_0, disp_map_0, acc_map_0, depth_map_0, depth_var_0 = rgb_map, disp_map, acc_map, depth_map, depth_var

            z_vals_mid = .5 * (z_vals[...,1:] + z_vals[...,:-1])
            z_samples = sample_pdf(z_vals_mid, weights[...,1:-1], self.config['training']['n_importance'], det=(self.config['training']['perturb']==0.))
            z_samples = z_samples.detach()

            z_vals, _ = torch.sort(torch.cat([z_vals, z_samples], -1), -1)
            pts = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None] # [N_rays, N_samples + N_importance, 3]

            raw = self.run_network(pts)
            rgb_map, disp_map, acc_map, weights, depth_map, depth_var = self.raw2outputs(raw, z_vals, self.config['training']['white_bkgd'])

        # F. Return rendering outputs, 返回一个字典，包含RGB图像、深度图、不透明度累积和深度方差等的结果
        ret = {'rgb' : rgb_map, 'depth' :depth_map, 
               'disp_map' : disp_map, 'acc_map' : acc_map, 
               'depth_var':depth_var,}
        ret = {**ret, 'z_vals': z_vals}

        ret['raw'] = raw

        # Importance sampling，yaml里默认是0，不执行
        if self.config['training']['n_importance'] > 0:
            ret['rgb0'] = rgb_map_0
            ret['disp0'] = disp_map_0
            ret['acc0'] = acc_map_0
            ret['depth0'] = depth_map_0
            ret['depth_var0'] = depth_var_0
            ret['z_std'] = torch.std(z_samples, dim=-1, unbiased=False)

        return ret
    
    def forward(self, rays_o, rays_d, target_rgb, target_d, global_step=0):
        '''
        Params:
            rays_o: ray origins (Bs, 3)
            rays_d: ray directions (Bs, 3)
            frame_ids: use for pose correction (Bs, 1)
            target_rgb: rgb value (Bs, 3)
            target_d: depth value (Bs, 1)
            c2w_array: poses (N, 4, 4) 
             r r r tx
             r r r ty
             r r r tz
        '''

        # 渲染函数，得到渲染的结果，传入参数是采样光线的起点和方向，以及深度值，传出的结果是一个字典，包含RGB图像、深度图、不透明度累积和深度方差等的结果
        rend_dict = self.render_rays(rays_o, rays_d, target_d=target_d)

        # 非训练流程里，则直接返回
        if not self.training:
            return rend_dict
        
        valid_depth_mask = (target_d.squeeze() > 0.) * (target_d.squeeze() < self.config['cam']['depth_trunc'])
        rgb_weight = valid_depth_mask.clone().unsqueeze(-1)
        rgb_weight[rgb_weight==0] = self.config['training']['rgb_missing']

        # ********************* 根据论文公式(6)计算颜色和深度的损失函数 *********************
        # 从rend_dict字典里取值
        rgb_loss = compute_loss(rend_dict["rgb"]*rgb_weight, target_rgb*rgb_weight)
        psnr = mse2psnr(rgb_loss)
        depth_loss = compute_loss(rend_dict["depth"].squeeze()[valid_depth_mask], target_d.squeeze()[valid_depth_mask])

        if 'rgb0' in rend_dict:
            rgb_loss += compute_loss(rend_dict["rgb0"]*rgb_weight, target_rgb*rgb_weight)
            depth_loss += compute_loss(rend_dict["depth0"][valid_depth_mask], target_d.squeeze()[valid_depth_mask])
        
        # ********************* 根据论文公式(7)和(8)计算sdf和free-space损失 *********************
        z_vals = rend_dict['z_vals']  # [N_rand, N_samples + N_importance]
        sdf = rend_dict['raw'][..., -1]  # [N_rand, N_samples + N_importance]
        truncation = self.config['training']['trunc'] * self.config['data']['sc_factor']
        fs_loss, sdf_loss = get_sdf_loss(z_vals, target_d, sdf, truncation, 'l2', grad=None)         
        

        ret = {
            "rgb": rend_dict["rgb"],
            "depth": rend_dict["depth"],
            "rgb_loss": rgb_loss,
            "depth_loss": depth_loss,
            "sdf_loss": sdf_loss,
            "fs_loss": fs_loss,
            "psnr": psnr,
        }

        # 返回一个字典，内含多个计算的loss结果
        return ret
