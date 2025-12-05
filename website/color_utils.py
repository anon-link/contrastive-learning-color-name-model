import numpy as np

from skimage.color import deltaE_ciede2000
import torch

def hex_to_rgb(hex_color):
    """将十六进制颜色转换为RGB值"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def rgb_255_to_01(r, g, b):
    """将0-255范围的RGB值转换为0-1范围"""
    return [r/255.0, g/255.0, b/255.0]

    
def differentiable_color_loss(lab1, lab2):
    """
    可导的颜色差异损失函数
    使用改进的L2距离替代不可导的CIEDE2000
    
    Args:
        lab1, lab2: torch.Tensor, shape (batch_size, 3) 或 (3,)
    
    Returns:
        torch.Tensor: 可导的色差损失
    """
    if lab1.dim() == 1:
        lab1 = lab1.unsqueeze(0)
    if lab2.dim() == 1:
        lab2 = lab2.unsqueeze(0)
    
    # 对L、a、b三个通道分别加权，模拟CIEDE2000的感知权重
    # L通道（亮度）通常更重要，权重更高
    # a、b通道（色度）权重相对较低
    l_weight, a_weight, b_weight = 1.0, 1.0, 1.0
    
    # 计算加权L2距离
    l_diff = l_weight * (lab1[:, 0] - lab2[:, 0]) ** 2
    a_diff = a_weight * (lab1[:, 1] - lab2[:, 1]) ** 2  
    b_diff = b_weight * (lab1[:, 2] - lab2[:, 2]) ** 2
    
    # 使用平滑的L2距离，避免在0点附近梯度消失
    epsilon = 1e-6
    color_diff = torch.sqrt(l_diff + a_diff + b_diff + epsilon)
    
    return color_diff


def custom_ciede2000(lab1, lab2):
    """
    使用skimage.color.deltaE_ciede2000计算CIEDE2000色差
    lab1, lab2: (3,) 或 (N,3) 的numpy数组
    返回: float
    """
    lab1 = np.array(lab1).reshape(1, 1, 3)
    lab2 = np.array(lab2).reshape(1, 1, 3)
    return float(deltaE_ciede2000(lab1, lab2)[0, 0])

    
def rgb_to_hsl(rgb):
    # rgb: [0,1]区间
    r, g, b = rgb
    max_val = max(r, g, b)
    min_val = min(r, g, b)
    diff = max_val - min_val
    
    if max_val == min_val:
        h = 0
    elif max_val == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif max_val == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360
    
    l = (max_val + min_val) / 2
    
    if max_val == min_val:
        s = 0
    elif l <= 0.5:
        s = diff / (max_val + min_val)
    else:
        s = diff / (2 - max_val - min_val)
    
    return [h/360, s, l]  # 归一化到[0,1]

def rgb_to_lab(rgb):
    """
    将RGB值转换为CIELAB颜色空间
    
    Args:
        rgb: RGB值数组或列表 [r, g, b]，范围0-1
    
    Returns:
        LAB值数组 [L, a, b]，其中：
        - L: 亮度，范围 [0, 100]
        - a: 红绿轴，范围 [-128, 127] 
        - b: 蓝黄轴，范围 [-128, 127]
    """
    r, g, b = rgb
    
    # 步骤1: RGB -> XYZ (使用sRGB标准)
    # 首先进行gamma校正
    def gamma_correct(c):
        if c <= 0.04045:
            return c / 12.92
        else:
            return ((c + 0.055) / 1.055) ** 2.4
    
    r_lin = gamma_correct(r)
    g_lin = gamma_correct(g)
    b_lin = gamma_correct(b)
    
    # sRGB到XYZ的转换矩阵 (D65白点)
    # 参考: https://en.wikipedia.org/wiki/SRGB#The_forward_transformation_(CIE_XYZ_to_sRGB)
    x = 0.4124564 * r_lin + 0.3575761 * g_lin + 0.1804375 * b_lin
    y = 0.2126729 * r_lin + 0.7151522 * g_lin + 0.0721750 * b_lin
    z = 0.0193339 * r_lin + 0.1191920 * g_lin + 0.9503041 * b_lin
    
    # 步骤2: XYZ -> LAB
    # D65白点参考值
    x_n, y_n, z_n = 0.95047, 1.00000, 1.08883
    
    def f(t):
        if t > (6/29)**3:
            return t ** (1/3)
        else:
            return (1/3) * (29/6)**2 * t + (4/29)
    
    # 计算LAB值
    l = 116 * f(y/y_n) - 16
    a = 500 * (f(x/x_n) - f(y/y_n))
    b = 200 * (f(y/y_n) - f(z/z_n))
    
    # 确保L在[0,100]范围内
    l = max(0, min(100, l))
    
    return [l, a, b]

def rgb_to_hcl(rgb):
    """
    将RGB值转换为HCL颜色空间
    
    Args:
        rgb: RGB值数组或列表 [r, g, b]，范围0-1
    
    Returns:
        HCL值数组 [H, C, L]，其中：
        - H: 色相，范围 [0, 1] (0度到360度归一化)
        - C: 色度，范围 [0, 1] (归一化)
        - L: 亮度，范围 [0, 1] (归一化)
    """
    # 先转lab
    lab = rgb_to_lab(rgb)
    l, a, b = lab
    
    # 计算色相 (Hue)
    h = np.arctan2(b, a) * 180 / np.pi
    if h < 0:
        h += 360
    h = h / 360  # 归一化到[0,1]
    
    # 计算色度 (Chroma)
    c = np.sqrt(a*a + b*b)
    # 色度通常不会超过约150，归一化到[0,1]
    c = min(1.0, c / 150)
    
    # 亮度归一化到[0,1]
    l = l / 100
    
    return [h, c, l]

def rgb_to_cmyk(rgb):
    # rgb: [0,1]
    r, g, b = rgb
    k = 1 - max(r, g, b)
    if k == 1:
        c = m = y = 0
    else:
        c = (1 - r - k) / (1 - k)
        m = (1 - g - k) / (1 - k)
        y = (1 - b - k) / (1 - k)
    return [c, m, y, k]

def rgb_to_16d_feature(rgb):
    """
    将RGB值转换为16维特征向量
    
    Args:
        rgb: RGB值数组或列表 [r, g, b]，范围0-1
    
    Returns:
        16维特征向量
    """
    # 确保rgb是列表格式
    if hasattr(rgb, 'tolist'):
        rgb_list = rgb.tolist()
    else:
        rgb_list = list(rgb)
    
    hsl = rgb_to_hsl(rgb_list)
    # H: 0-360 -> 0-1, S/L: 0-1
    hsl_norm = [hsl[0] / 360.0, hsl[1], hsl[2]]
    lab = rgb_to_lab(rgb_list)
    # L: 0-100 -> 0-1, a: -128~127 -> 0-1, b: -128~127 -> 0-1
    lab_norm = [
        lab[0] / 100.0,
        (lab[1] + 128) / 255.0,
        (lab[2] + 128) / 255.0
    ]
    hcl = rgb_to_hcl(rgb_list)
    # H: 0-360 -> 0-1, C/L: 0-1
    hcl_norm = [hcl[0] / 360.0, hcl[1], hcl[2]]
    cmyk = rgb_to_cmyk(rgb_list)
    # C/M/Y/K: 0-1
    cmyk_norm = [cmyk[0], cmyk[1], cmyk[2], cmyk[3]]
    
    # 拼接为16维
    feature = (
                [rgb_list[0], rgb_list[1], rgb_list[2]] +
                [hsl_norm[0], hsl_norm[1], hsl_norm[2]] +
                [lab_norm[0], lab_norm[1], lab_norm[2]] +
                [hcl_norm[0], hcl_norm[1], hcl_norm[2]] +
                [cmyk_norm[0], cmyk_norm[1], cmyk_norm[2], cmyk_norm[3]]
            )
    return feature
    
def rgb_to_lab_torch(rgb_tensor):
    """
    可微分的RGB到LAB转换函数（PyTorch版本）
    
    Args:
        rgb_tensor: RGB tensor，形状为 [batch_size, 3]，值范围 [0, 1]
    
    Returns:
        lab_tensor: LAB tensor，形状为 [batch_size, 3]
    """
    # 确保输入在[0,1]范围内
    rgb_tensor = torch.clamp(rgb_tensor, 0, 1)
    
    # 步骤1: RGB -> XYZ (使用sRGB标准)
    # 首先进行gamma校正
    def gamma_correct_torch(c):
        return torch.where(c <= 0.04045, c / 12.92, torch.pow((c + 0.055) / 1.055, 2.4))
    
    r, g, b = rgb_tensor[:, 0], rgb_tensor[:, 1], rgb_tensor[:, 2]
    
    # 应用gamma校正
    r_linear = gamma_correct_torch(r)
    g_linear = gamma_correct_torch(g)
    b_linear = gamma_correct_torch(b)
    
    # 转换到XYZ颜色空间（使用sRGB矩阵）
    # 这些是sRGB到XYZ的转换矩阵
    x = 0.4124564 * r_linear + 0.3575761 * g_linear + 0.1804375 * b_linear
    y = 0.2126729 * r_linear + 0.7151522 * g_linear + 0.0721750 * b_linear
    z = 0.0193339 * r_linear + 0.1191920 * g_linear + 0.9503041 * b_linear
    
    # 步骤2: XYZ -> LAB
    # 使用D65白点 (0.95047, 1.00000, 1.08883)
    xn, yn, zn = 0.95047, 1.00000, 1.08883
    
    # 归一化
    fx = x / xn
    fy = y / yn
    fz = z / zn
    
    # 应用LAB转换函数
    def lab_f_torch(t):
        delta = 6.0 / 29.0
        return torch.where(t > delta**3, torch.pow(t, 1.0/3.0), t / (3 * delta**2) + 4.0/29.0)
    
    fx_lab = lab_f_torch(fx)
    fy_lab = lab_f_torch(fy)
    fz_lab = lab_f_torch(fz)
    
    # 计算LAB值
    L = 116 * fy_lab - 16
    a = 500 * (fx_lab - fy_lab)
    b = 200 * (fy_lab - fz_lab)
    
    # 组合结果
    lab_tensor = torch.stack([L, a, b], dim=1)
    
    return lab_tensor
