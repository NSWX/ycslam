import torch
import numpy as np
import tinycudann as tcnn


def get_encoder(encoding, input_dim=3,
                degree=4, n_bins=16, n_frequencies=12,
                n_levels=16, level_dim=2, 
                base_resolution=16, log2_hashmap_size=19, 
                desired_resolution=512):
    '''        
    创建不同类型的编码器（encoder），通过配置不同的参数生成用于神经网络输入的数据编码
    Args:
        encoding: 编码器类型的字符串，用于指定你希望使用的编码方式。
        input_dim: 输入维度，通常表示数据的维度。例如，对于3D坐标，输入维度为3。
        degree: 对于球谐函数编码（spherical），这是球谐函数的阶数。
                阶数越高，函数的复杂度越大，能够表示更复杂的方向分布。
        n_bins: 对于Blob编码（blob），这是分箱的数量。
                分箱的数量决定了编码器将输入空间划分为多少个离散区间。
        n_frequencies: 对于频率编码（freq），这是频率的数量。
                频率编码通过在输入上应用不同频率的正弦和余弦函数来捕捉输入的细节特征。。
        n_levels: 这是网格的层数。
                层数越多，编码器能够表示的细节越丰富。每个层级表示不同分辨率下的空间特征。
        level_dim: 每一层的特征维度，表示在每个层级上为每个点编码多少个特征值。
        base_resolution: 网格编码的基础分辨率。
                        它是最低层级的分辨率，表示在最粗糙的网格中，空间被划分为多少个单元。
        log2_hashmap_size: 对于稀疏网格编码（hash），这是哈希表大小的对数基2值。
                        哈希表用于稀疏地存储网格数据。值越大，哈希表越大，能够表示的空间特征也越多。
        desired_resolution: 目标分辨率，表示编码器在最高层级上应能够表示的空间分辨率。
                        这与base_resolution和n_levels一起决定了每层的分辨率增长速度。
    Returns:
        embed: 创建的编码器。
        out_dim: 输出维度。

    '''    
    
    #! 密集网格编码 (Dense Grid Encoding)
    if 'dense' in encoding.lower(): #? 如果编码类型包含 'dense'，则创建一个密集网格编码器。
        n_levels = 4
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1)) # 用于计算每层的尺度
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                    "otype": "Grid",
                    "type": "Dense",
                    "n_levels": n_levels,
                    "n_features_per_level": level_dim,
                    "base_resolution": base_resolution,
                    "per_level_scale": per_level_scale,
                    "interpolation": "Linear"}, #! 使用线性插值
                dtype=torch.float
        )
        out_dim = embed.n_output_dims
    
    #! 稀疏网格编码（Sparse Grid Encoding）
    elif 'hash' in encoding.lower() or 'tiled' in encoding.lower(): #? 如果编码类型包含 'hash' 或 'tiled'，则创建一个稀疏网格编码器。
        print('Hash size', log2_hashmap_size)
        per_level_scale = np.exp2(np.log2(desired_resolution  / base_resolution) / (n_levels - 1))
        embed = tcnn.Encoding(
            n_input_dims=input_dim,
            encoding_config={
                "otype": 'HashGrid',  #! 允许在较大分辨率下使用更少的存储空间
                "n_levels": n_levels,
                "n_features_per_level": level_dim,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": per_level_scale
            },
            dtype=torch.float
        )
        out_dim = embed.n_output_dims

    #! 球谐函数编码（Spherical Harmonics Encoding）
    elif 'spherical' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "SphericalHarmonics",
                "degree": degree,
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    #! Blob编码（OneBlob Encoding）
    elif 'blob' in encoding.lower():
        print('Use blob')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "OneBlob", # Component type.
	            "n_bins": n_bins    # n_bins 指定分箱数量
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    #! 频率编码（Frequency Encoding）
    elif 'freq' in encoding.lower():
        print('Use frequency')
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Frequency", 
                "n_frequencies": n_frequencies # n_frequencies 指定频率数量
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims
    
    #! 恒等编码（Identity Encoding）
    elif 'identity' in encoding.lower():
        embed = tcnn.Encoding(
                n_input_dims=input_dim,
                encoding_config={
                "otype": "Identity"
                },
                dtype=torch.float
            )
        out_dim = embed.n_output_dims

    return embed, out_dim