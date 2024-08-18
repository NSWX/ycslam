# Package imports
import torch
import torch.nn as nn
import tinycudann as tcnn


class ColorNet(nn.Module):
    """
    ColorNet 类用于从几何特征和输入数据中预测 RGB 颜色值。

    Attributes:
        - config (dict): 包含网络配置和其他参数的配置字典。
        - input_ch (int): 输入通道的数量，包括方向编码和其他特征。
        - geo_feat_dim (int): 几何特征维度。
        - hidden_dim_color (int): 隐藏层的维度，用于控制网络的容量。
        - num_layers_color (int): 颜色网络中的层数。
        - model (nn.Module): 实际的颜色预测模型，可能是 tcnn 网络或传统的多层感知器（MLP）。
    """
    def __init__(self, config, input_ch=4, geo_feat_dim=15, 
                hidden_dim_color=64, num_layers_color=3):
        """
        初始化 ColorNet 类。

        Args:
            - config (dict): 包含网络配置参数的字典。
            - input_ch (int): 输入特征的维度。 默认：4
            - geo_feat_dim (int): 几何特征的维度。 默认：15
            - hidden_dim_color (int): 隐藏层的维度。默认：64
            - num_layers_color (int): 网络层的数量。默认：3
        """
        super(ColorNet, self).__init__()
        self.config = config
        self.input_ch = input_ch                    # 48
        self.geo_feat_dim = geo_feat_dim            # 15 几何特征维度
        self.hidden_dim_color = hidden_dim_color    # 32
        self.num_layers_color = num_layers_color    # 2

        self.model = self.get_model(config['decoder']['tcnn_network'])
    
    def forward(self, input_feat):
        """
        前向传播函数，将输入特征传递给模型进行颜色预测。

        Args:
            - input_feat (torch.Tensor): 输入特征张量。

        Returns:
            - torch.Tensor: 模型输出的颜色预测结果。
        """
        # h = torch.cat([embedded_dirs, geo_feat], dim=-1)
        return self.model(input_feat)
    
    def get_model(self, tcnn_network=False):
        """
        根据配置选择模型类型（使用 TinyCUDA 或传统的 PyTorch 模型）。

        Args:
            - tcnn_network (bool): 是否使用 TinyCUDA 网络。

        Returns:
            - nn.Module: 定义的神经网络模型。
        """
        if tcnn_network:
            print('Color net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch + self.geo_feat_dim,
                n_output_dims=3,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim_color,
                    "n_hidden_layers": self.num_layers_color - 1,
                },
                #dtype=torch.float
            )

        color_net =  []
        for l in range(self.num_layers_color):
            if l == 0:
                in_dim = self.input_ch + self.geo_feat_dim
            else:
                in_dim = self.hidden_dim_color
            
            if l == self.num_layers_color - 1:
                out_dim = 3 # 3 rgb
            else:
                out_dim = self.hidden_dim_color
            
            color_net.append(nn.Linear(in_dim, out_dim, bias=False))
            if l != self.num_layers_color - 1:
                color_net.append(nn.ReLU(inplace=True))

        return nn.Sequential(*nn.ModuleList(color_net))

class SDFNet(nn.Module):
    """
    SDFNet 类用于生成签名距离场 (SDF) 预测的神经网络模型。

    Attributes:
        - config (dict): 包含网络配置参数的字典。
        - input_ch (int): 输入特征的维度。
        - geo_feat_dim (int): 几何特征的维度。
        - hidden_dim (int): 隐藏层的维度。
        - num_layers (int): 网络层的数量。
        - model (nn.Module): 定义的神经网络模型。
    """
    def __init__(self, config, input_ch=3, geo_feat_dim=15, hidden_dim=64, num_layers=2):
        """
        初始化 SDFNet 类。

        Args:
            - config (dict): 包含网络配置参数的字典。
            - input_ch (int): 输入特征的维度。
            - geo_feat_dim (int): 几何特征的维度。
            - hidden_dim (int): 隐藏层的维度。
            - num_layers (int): 网络层的数量。
        """
        super(SDFNet, self).__init__()
        self.config = config
        self.input_ch = input_ch
        self.geo_feat_dim = geo_feat_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.model = self.get_model(tcnn_network=config['decoder']['tcnn_network'])
    
    def forward(self, x, return_geo=True):
        """
        前向传播函数，将输入特征传递给模型进行 SDF 预测。

        Args:
            - x (torch.Tensor): 输入特征张量。
            - return_geo (bool): 是否返回几何特征。

        Returns:
            - torch.Tensor: 模型输出的 SDF 和几何特征。
        """
        out = self.model(x)

        if return_geo:  # return feature
            return out
        else:
            return out[..., :1]

    def get_model(self, tcnn_network=False):
        """
        根据配置选择模型类型（使用 TinyCUDA 或传统的 PyTorch 模型）。

        Args:
            - tcnn_network (bool): 是否使用 TinyCUDA 网络。

        Returns:
            - nn.Module: 定义的神经网络模型。
        """
        if tcnn_network:
            print('SDF net: using tcnn')
            return tcnn.Network(
                n_input_dims=self.input_ch,
                n_output_dims=1 + self.geo_feat_dim,
                network_config={
                    "otype": "FullyFusedMLP",
                    "activation": "ReLU",
                    "output_activation": "None",
                    "n_neurons": self.hidden_dim,
                    "n_hidden_layers": self.num_layers - 1,
                },
                #dtype=torch.float
            )
        else:
            sdf_net = []
            for l in range(self.num_layers):
                if l == 0:
                    in_dim = self.input_ch
                else:
                    in_dim = self.hidden_dim 
                
                if l == self.num_layers - 1:
                    out_dim = 1 + self.geo_feat_dim # 1 sigma + 15 SH features for color
                else:
                    out_dim = self.hidden_dim 
                
                sdf_net.append(nn.Linear(in_dim, out_dim, bias=False))
                if l != self.num_layers - 1:
                    sdf_net.append(nn.ReLU(inplace=True))

            return nn.Sequential(*nn.ModuleList(sdf_net))

class ColorSDFNet(nn.Module):
    '''
    ColorSDFNet 类用于同时处理颜色和 SDF 预测，结合了颜色网络和 SDF 网络。

    Attributes:
        - config (dict): 包含网络配置参数的字典。
        - color_net (ColorNet): 颜色网络模型实例。
        - sdf_net (SDFNet): SDF 网络模型实例。
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        """
        初始化 ColorSDFNet 类。

        Args:
            - config (dict): 包含网络配置参数的字典。
            - input_ch (int): 输入特征的维度。
            - input_ch_pos (int): 位置编码的维度。
        """
        super(ColorSDFNet, self).__init__()
        self.config = config
        self.color_net = ColorNet(config, 
                input_ch=input_ch+input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])

    # 允许颜色网络直接访问原始的颜色嵌入信息(ColorSDFNet的特点)        
    def forward(self, embed, embed_pos, embed_color):
        """
        前向传播函数，计算颜色和 SDF 预测。

        Args:
            - embed (torch.Tensor): 输入特征张量。
            - embed_pos (torch.Tensor): 位置编码特征张量。
            - embed_color (torch.Tensor): 颜色特征张量。

        Returns:
            - torch.Tensor: 拼接的 RGB 颜色和 SDF 结果。
        """

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, embed_color, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([embed_color, geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)
    
class ColorSDFNet_v2(nn.Module):
    '''
    No color grid
    ColorSDFNet_v2 类用于同时处理颜色和 SDF 预测，不直接使用独立的颜色网格，而是通过空间编码和几何特征生成颜色信息。

    Attributes:
        - config (dict): 包含网络配置参数的字典。
        - color_net (ColorNet): 颜色网络模型实例。
        - sdf_net (SDFNet): SDF 网络模型实例。
    '''
    def __init__(self, config, input_ch=3, input_ch_pos=12):
        """
        初始化 ColorSDFNet_v2 类。

        Args:
            - config (dict): 包含网络配置参数的字典。
            - input_ch (int): 输入特征的维度。
            - input_ch_pos (int): 位置编码的维度。
        """
        super(ColorSDFNet_v2, self).__init__()
        self.config = config
        self.color_net = ColorNet(config, 
                input_ch=input_ch_pos, 
                geo_feat_dim=config['decoder']['geo_feat_dim'], 
                hidden_dim_color=config['decoder']['hidden_dim_color'], 
                num_layers_color=config['decoder']['num_layers_color'])
        self.sdf_net = SDFNet(config,
                input_ch=input_ch+input_ch_pos,
                geo_feat_dim=config['decoder']['geo_feat_dim'],
                hidden_dim=config['decoder']['hidden_dim'], 
                num_layers=config['decoder']['num_layers'])
            
    # 颜色信息是通过结合空间编码和几何特征来生成的，而不是直接从独立的颜色数据中提取(ColorSDFNet_v2的特点)
    def forward(self, embed, embed_pos):
        """
        前向传播函数，计算颜色和 SDF 预测。

        Args:
            - embed (torch.Tensor): 输入特征张量。
            - embed_pos (torch.Tensor): 位置编码特征张量。

        Returns:
            - torch.Tensor: 拼接的 RGB 颜色和 SDF 结果。
        """

        if embed_pos is not None:
            h = self.sdf_net(torch.cat([embed, embed_pos], dim=-1), return_geo=True) 
        else:
            h = self.sdf_net(embed, return_geo=True) 
        
        sdf, geo_feat = h[...,:1], h[...,1:]
        if embed_pos is not None:
            rgb = self.color_net(torch.cat([embed_pos, geo_feat], dim=-1))
        else:
            rgb = self.color_net(torch.cat([geo_feat], dim=-1))
        
        return torch.cat([rgb, sdf], -1)
