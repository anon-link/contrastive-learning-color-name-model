import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from color_utils import *
import time
from collections import Counter
import logging
from datetime import datetime
import sys
import csv

# 预训练Transformer相关导入
from transformers import AutoTokenizer, AutoModel
import warnings
warnings.filterwarnings('ignore')

if torch.cuda.device_count() > 1:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# NCF架构配置
use_ncf_architecture = True  # 是否使用NCF架构
used_learning_rate = 1e-4
used_num_epochs = 30
use_for_test = False
use_16d_feature = True
use_freezed_weight = True
use_high_freq_data = False
use_lab_dist = False
use_negtive_sample = True

# test [1,0,0], [0,1,0], [1,1,0], [0,1,1] [1,0,1], [1,1,1]
setting = [0,1,1,0]
contrastive_loss_weight_arr = [0, 1]
binary_loss_weight_arr = [0, 1]
color_distance_weight_arr = [0, 1]
color_embedding_weight_arr = [0, 1]
contrastive_loss_weight = contrastive_loss_weight_arr[setting[0]]
binary_loss_weight = binary_loss_weight_arr[setting[1]]
color_distance_weight = color_distance_weight_arr[setting[2]]
color_embedding_weight = color_embedding_weight_arr[setting[3]]
rgb_generator_hidden_dims=[]
add_name_encoder = 1
device_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')

now_date = datetime.now().strftime("%m%d%H%M%S")
suffix_str = "_0905_test_"+str(use_for_test)+"_high_" + str(use_high_freq_data)+"_use16d_" + str(use_16d_feature) + "_weight_" + "".join(str(x) for x in setting) + "_ncf_" + str(binary_loss_weight) + "_hidden_" + "_".join(str(x) for x in rgb_generator_hidden_dims)+"_epoch_" + str(used_num_epochs) + "_device_" + str(device_id)
# suffix_str = "_0903_test_False_high_False_use16d_False_weight_0110_ncf_0.01_hidden__device_1"
suffix_str = f"_{now_date}"
use_ab_data_weight = 0
# if len(sys.argv) > 1:
#     binary_loss_weight = float(sys.argv[2])
#     used_num_epochs = int(sys.argv[3])
#     add_name_encoder = int(sys.argv[4])
#     os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[5]
#     device_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
#     use_for_test = sys.argv[6].lower() == "true"
#     use_ab_data_weight = float(sys.argv[7])
#     use_high_freq_data = sys.argv[8].lower() == "true"
#     use_freezed_weight = sys.argv[9].lower() == "true"
#     now_date = datetime.now().strftime("%m%d")
#     # now_date = "0910"
#     suffix_str = f"_{now_date}_test_{use_for_test}_ncf_{binary_loss_weight}_generator_{color_distance_weight}_epoch_{used_num_epochs}_device_{device_id}_name_{add_name_encoder}_ab_{use_ab_data_weight}_high_{use_high_freq_data}_freezed_{use_freezed_weight}"
#     if not use_negtive_sample:
#         suffix_str += "_negtive_False"
#     if not use_ncf_architecture:
#         suffix_str += "_ncf_False"
#     if not use_16d_feature:
#         suffix_str += "_16d_False"

# 配置日志记录
# 检查是否为评估模式
is_evaluation_mode = len(sys.argv) > 1 and sys.argv[1] == "evaluate"

if is_evaluation_mode:
    log_filename = f"evaluate{suffix_str}.log"
else:
    # 训练模式：同时输出到控制台和文件
    log_filename = f"train{suffix_str}.log"
# 如果日志文件已存在，则覆盖（即直接写入，不追加）
if os.path.exists(log_filename):
    os.remove(log_filename)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(log_filename, encoding='utf-8'),  # 保存到文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"日志文件: {log_filename}")
logger.info(f"starting time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")
logger.info(f"当前设置：{suffix_str}")
logger.info(f"设备: {device_id}")
logger.info(f"小样本训练: {use_for_test}")
logger.info(f"使用16d特征: {use_16d_feature}")
logger.info(f"冻结预训练权重: {use_freezed_weight}")
logger.info(f"使用NCF架构: {use_ncf_architecture}")
logger.info(f"使用负样本: {use_negtive_sample}")
logger.info(f"使用高频数据: {use_high_freq_data}")
logger.info(f"使用LAB距离: {use_lab_dist}")
logger.info(f"RGB生成器隐藏层维度: {rgb_generator_hidden_dims}")
logger.info(f"对比损失权重: {contrastive_loss_weight}")
logger.info(f"分类损失权重: {binary_loss_weight}")
logger.info(f"颜色距离权重: {color_distance_weight}")
logger.info(f"颜色embedding权重: {color_embedding_weight}")
logger.info(f"学习率: {used_learning_rate}")
logger.info(f"训练轮数: {used_num_epochs}")


# 设置随机种子
# torch.manual_seed(42)
# np.random.seed(42)

import matplotlib
# 设置中文字体，优先使用系统可用字体
import platform
if platform.system() == 'Windows':
    # Windows系统字体
    font_list = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
elif platform.system() == 'Darwin':
    # macOS系统字体
    font_list = ['PingFang SC', 'Hiragino Sans GB', 'STHeiti', 'Arial Unicode MS']
else:
    # Linux系统字体
    font_list = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS']

# 设置字体
matplotlib.rcParams['font.sans-serif'] = font_list
matplotlib.rcParams['axes.unicode_minus'] = False    # 正常显示负号

# 如果仍然找不到字体，使用默认字体
try:
    import matplotlib.font_manager as fm
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    logger.info(f"可用字体数量: {len(available_fonts)}")
    logger.info(f"前10个字体: {available_fonts[:10]}")
except Exception as e:
    logger.error(f"字体检查失败: {e}")
# -------------------------
# 1. 数据预处理
# -------------------------
def preprocess_data(csv_path='responses.csv', min_count=10):
    """
    预处理颜色名称数据
    
    Args:
        csv_path: CSV文件路径
        min_count: 最小词频阈值
    
    Returns:
        rgb_values: 归一化的RGB值
        encoded_labels: 编码后的标签
        label_encoder: 标签编码器
        unique_terms: 唯一颜色名称列表
    """
    logger.info("正在加载数据...")
    df = pd.read_csv(csv_path)
    logger.info(f"原始数据: {len(df)} 行")
    pre_df_rn = len(df)
    
    pre_term_counts = df['term'].value_counts()
    logger.info(f"唯一颜色名称: {len(pre_term_counts)} 个")
    pre_unique_rgb = df[['r', 'g', 'b']].drop_duplicates()
    logger.info(f"唯一RGB值: {len(pre_unique_rgb)} 个")

    # 将df中term字符数少于3个的行删除
    df = df[df['term'].str.len() >= 3].reset_index(drop=True)
    logger.info(f"删除term字符数少于3个的行后，数据: {len(df)} 行, 删除了 {pre_df_rn - len(df)} 行")

    term_counts = df['term'].value_counts()
    logger.info(f"唯一颜色名称: {len(term_counts)} 个, 删除了 {len(pre_term_counts) - len(term_counts)} 个")

    # 统计前10颜色名称及出现次数，并统计每个名称出现最多次的RGB值
    top10_terms = term_counts.head(10)
    logger.info("前10个颜色名称及出现次数及其最常见的RGB值:")
    for term, count in top10_terms.items():
        # 找到该term对应的所有RGB
        term_rgbs = df[df['term'] == term][['r', 'g', 'b']]
        # 统计出现最多的RGB
        rgb_mode = term_rgbs.value_counts().idxmax()
        rgb_mode_count = term_rgbs.value_counts().max()
        logger.info(f"  {term}: {count}，最常见RGB: {rgb_mode}，出现次数: {rgb_mode_count}")
    
    # 统计唯一RGB值 - 将r,g,b三列组合成RGB元组
    unique_rgb = df[['r', 'g', 'b']].drop_duplicates()
    logger.info(f"唯一RGB值: {len(unique_rgb)} 个, 删除了 {len(pre_unique_rgb) - len(unique_rgb)} 个")
    # high_freq_terms = term_counts[term_counts > min_count].index
    # df = df[df['term'].isin(high_freq_terms)].reset_index(drop=True)
    # logger.info(f"过滤后数据: {len(df)} 行")
    # logger.info(f"唯一颜色名称: {len(high_freq_terms)} 个")

    # 随机从df中取11w个样本
    if use_for_test:
        df = df.sample(n=60000, random_state=42).reset_index(drop=True)
        logger.info(f"已随机采样当前数据: {len(df)} 行")

    # 提取RGB值并归一化
    rgb_values = df[['r', 'g', 'b']].values.astype(np.float32) / 255.0
    # 获取rgb_values中所有的唯一RGB值
    if rgb_values.ndim == 2 and rgb_values.shape[1] >= 3:
        # 只取前3列作为RGB
        rgb_3d = rgb_values[:, :3]
        # 使用numpy去重
        unique_colors = np.unique(rgb_3d, axis=0)
        logger.info(f"从rgb_values中提取到 {len(unique_colors)} 个唯一RGB值")
        logger.info(f"前5个唯一RGB值示例: {unique_colors[:5]}")
    else:
        unique_colors = np.array([])
        logger.info("rgb_values不是2D数组或列数不足3，无法提取唯一RGB值")

        
    # 编码标签
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(df['term'].values)
    unique_terms = label_encoder.classes_
    
    logger.info(f"RGB值形状: {rgb_values.shape}")
    logger.info(f"编码标签形状: {encoded_labels.shape}")
    
    return rgb_values, encoded_labels, label_encoder, unique_terms, unique_colors

# -------------------------
# 2. 模型定义
# -------------------------

class NCFModel(nn.Module):
    """
    集成NCF架构的颜色名称模型
    在原有encoder基础上添加MLP层和特征融合机制
    """
    def __init__(self, emb_dim=64):
        super().__init__()
        self.emb_dim = emb_dim
        
        fusion_input_dim = emb_dim * 2
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, 256),  
            nn.ReLU(),
            nn.Linear(256, 128),  
            nn.ReLU(),
            nn.Linear(128, 1),  # 输出1维分数
            nn.Sigmoid()  # 确保输出在0-1之间
        )
    
    def forward(self, rgb_input, name_input):
        """
        Args:
            rgb_input: RGB特征输入 (emb_dim维)
            name_input: 名称输入 (emb_dim维)
        
        Returns:
            similarity_score: 相似度分数 (0-1之间)
        """
        
        # 特征融合
        fusion_input = torch.cat([rgb_input, name_input], dim=-1)  # emb_dim*2维
        # 通过融合MLP：fusion_input_dim -> 128维 -> 1维（0-1分数）
        similarity_score = self.fusion_mlp(fusion_input)  # 输出0-1之间的分数
        
        return similarity_score

class RGBGenerator(nn.Module):
    """
    RGB颜色值生成器
    接收名称编码器的输出，生成对应的RGB颜色值
    """
    def __init__(self, input_dim=64, hidden_dims=[], output_dim=3):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 构建MLP网络
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU()
            ])
            prev_dim = hidden_dim
        
        # 输出层：生成RGB值 (0-1范围)
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Sigmoid())  # 确保输出在0-1之间
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, name_embedding):
        """
        Args:
            name_embedding: 名称编码器的输出 [batch_size, input_dim]
        
        Returns:
            rgb_output: 生成的RGB值 [batch_size, 3]
        """
        return self.mlp(name_embedding)

class RGBEncoder(nn.Module):
    def __init__(self, emb_dim=64, input_dim=16):
        super().__init__()
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        logger.info(f"RGBEncoder input_dim: {input_dim}, emb_dim: {emb_dim}")
        self._create_network()
        

    def _create_network(self):
        self._build_feature_net(self.input_dim)
    
    def _build_feature_net(self, input_dim):
        """构建特征网络"""
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 128),  # 输入层：任意维度 → 128维
            nn.ReLU(),                  # 激活函数
            # nn.Dropout(0.2),            # 防止过拟合
            nn.Linear(128, 256),        # 隐藏层：128 → 256
            nn.ReLU(),
            # nn.Dropout(0.2),
            nn.Linear(256, self.emb_dim)     # 输出层：256 → emb_dim维嵌入
        )
    
    def forward(self, x):
        # 输出嵌入向量
        embedding = F.normalize(self.feature_net(x), dim=-1)
        return embedding

class NameEncoder(nn.Module):
    """
    基于预训练Transformer的颜色名称编码器
    在预训练模型基础上继续训练，获得更好的语义理解
    """
    def __init__(self, model_name='bert-base-uncased', emb_dim=64, max_length=32, 
                 freeze_pretrained=False, use_pooling=True, local_model_path=None, add_name_encoder=1):
        super().__init__()
        self.emb_dim = emb_dim
        self.max_length = max_length
        self.use_pooling = use_pooling
        
        # 加载预训练的tokenizer和模型
        logger.info(f"加载预训练模型: {model_name}")
        
        # 优先使用本地模型路径
        if local_model_path and os.path.exists(local_model_path):
            logger.info(f"使用本地模型: {local_model_path}")
            self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
            
            # 尝试加载模型，支持多种格式
            try:
                # 首先尝试从safetensors加载（你的模型格式）
                logger.info("尝试从safetensors加载模型...")
                self.pretrained_model = AutoModel.from_pretrained(
                    local_model_path, 
                    use_safetensors=True
                )
                logger.info("✅ 成功从safetensors加载模型")
            except Exception as e1:
                logger.info(f"safetensors加载失败: {e1}")
                try:
                    # 尝试标准加载
                    logger.info("尝试标准加载...")
                    self.pretrained_model = AutoModel.from_pretrained(local_model_path)
                    logger.info("✅ 成功加载模型（标准格式）")
                except Exception as e2:
                    logger.info(f"标准加载失败: {e2}")
                    try:
                        # 尝试从Flax权重加载
                        logger.info("尝试从Flax权重加载...")
                        self.pretrained_model = AutoModel.from_pretrained(
                            local_model_path,
                            from_flax=True
                        )
                        logger.info("✅ 成功从Flax权重加载模型")
                    except Exception as e3:
                        logger.info(f"所有本地加载方法都失败了: {e3}")
                        logger.info("尝试从网络下载模型...")
                        self.pretrained_model = AutoModel.from_pretrained(model_name)
        else:
            # 尝试从本地缓存加载
            try:
                logger.info("尝试从本地缓存加载模型...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
                
                # 尝试加载模型，支持多种格式
                try:
                    # 首先尝试标准加载
                    self.pretrained_model = AutoModel.from_pretrained(model_name, local_files_only=True)
                    logger.info("成功从本地缓存加载模型（标准格式）")
                except Exception as e1:
                    logger.info(f"标准加载失败: {e1}")
                    try:
                        # 尝试从safetensors加载
                        self.pretrained_model = AutoModel.from_pretrained(
                            model_name, 
                            local_files_only=True,
                            use_safetensors=True
                        )
                        logger.info("成功从本地缓存加载模型（safetensors格式）")
                    except Exception as e2:
                        logger.info(f"safetensors加载失败: {e2}")
                        raise e2
                        
            except Exception as e:
                logger.info(f"本地缓存加载失败: {e}")
                logger.info("尝试从网络下载模型...")
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                
                # 网络下载时也尝试多种格式
                try:
                    self.pretrained_model = AutoModel.from_pretrained(model_name)
                    logger.info("成功从网络下载模型（标准格式）")
                except Exception as e1:
                    logger.info(f"标准下载失败: {e1}")
                    try:
                        self.pretrained_model = AutoModel.from_pretrained(
                            model_name,
                            use_safetensors=True
                        )
                        logger.info("成功从网络下载模型（safetensors格式）")
                    except Exception as e2:
                        logger.info(f"safetensors下载失败: {e2}")
                        # 最后尝试
                        self.pretrained_model = AutoModel.from_pretrained(model_name)
        
        # 如果tokenizer没有pad_token，设置为eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if freeze_pretrained:
            for param in self.pretrained_model.parameters():
                param.requires_grad = False
            logger.info("预训练模型参数已冻结")
        else:
            logger.info("预训练模型参数可训练，将在您的数据上继续训练")
        
        # 输出投影层，将预训练模型的输出维度映射到目标维度
        pretrained_dim = self.pretrained_model.config.hidden_size
        # 构建MLP网络
        layers = []
        layers.append(nn.Linear(pretrained_dim, emb_dim * 2))
        
        for i in range(add_name_encoder):
            layers.extend([
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(emb_dim * 2, emb_dim * 2)
            ])
        
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(0.1))
        layers.append(nn.Linear(emb_dim * 2, emb_dim))
        
        self.output_projection = nn.Sequential(*layers)

        # 可选的注意力池化层
        if use_pooling:
            self.attention_pooling = nn.MultiheadAttention(
                embed_dim=emb_dim, 
                num_heads=4, 
                batch_first=True
            )
            # 使用可学习的查询参数，而不是每次都随机生成
            # 这样可以在训练过程中学习到稳定的查询表示
            self.query = nn.Parameter(torch.randn(1, 1, emb_dim))
            logger.info("✅ 使用可学习的注意力查询参数")
        else:
            logger.info("⚠️ 未使用注意力池化，将使用CLS token或平均池化")
    
    def to(self, device):
        """重写to方法，确保所有组件都移动到指定设备"""
        super().to(device)
        self._device = device
        # 确保预训练模型也移动到指定设备
        self.pretrained_model = self.pretrained_model.to(device)
        # 如果使用注意力池化，确保查询参数也在正确设备上
        if hasattr(self, 'query'):
            self.query = self.query.to(device)
        return self
    
    def forward(self, color_names):
        """
        Args:
            color_names: 颜色名称列表，如 ['red', 'light blue', 'dark crimson']
        
        Returns:
            embeddings: 颜色名称的嵌入向量
        """
        # 确保输入是字符串列表
        if isinstance(color_names, str):
            color_names = [color_names]
        elif not isinstance(color_names, list):
            raise ValueError(f"color_names must be a string or list of strings, got {type(color_names)}")
        
        # 确保所有元素都是字符串
        color_names = [str(name) for name in color_names]
        
        # 使用预训练tokenizer处理输入
        inputs = self.tokenizer(
            color_names,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 将输入移到正确的设备
        # 优先使用模型的设备，如果模型还没有被移动到设备上，则使用当前设备
        if hasattr(self, '_device') and self._device is not None:
            device = self._device
        else:
            device = next(self.pretrained_model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # 通过预训练模型（允许梯度传播）
        outputs = self.pretrained_model(**inputs)
        
        # 获取最后一层的隐藏状态
        hidden_states = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)
        
        # 通过输出投影层
        projected = self.output_projection(hidden_states)  # (batch_size, seq_len, emb_dim)
        
        if self.use_pooling:
            # 使用注意力池化
            # 使用可学习的查询参数，确保一致性
            batch_size = projected.size(0)
            # 将可学习的查询参数扩展到当前批次大小
            query = self.query.expand(batch_size, 1, self.emb_dim)
            
            # 注意力池化
            pooled, _ = self.attention_pooling(query, projected, projected)
            pooled = pooled.squeeze(1)  # (batch, emb_dim)
        else:
            # 使用CLS token或平均池化
            if 'bert' in self.pretrained_model.config.model_type:
                # BERT使用CLS token
                pooled = projected[:, 0, :]  # 取第一个token (CLS)
            else:
                # 其他模型使用平均池化
                # 创建注意力掩码
                attention_mask = inputs['attention_mask'].unsqueeze(-1)  # (batch_size, seq_len, 1)
                masked_projected = projected * attention_mask
                pooled = masked_projected.sum(dim=1) / attention_mask.sum(dim=1)  # 平均池化
        
        # 归一化
        return F.normalize(pooled, dim=-1)
    
    def encode_single(self, color_name):
        """编码单个颜色名称"""
        return self.forward([color_name]).squeeze(0)
    
    def encode_batch(self, color_names):
        """编码一批颜色名称"""
        # 确保输入是列表
        if not isinstance(color_names, list):
            raise ValueError(f"encode_batch expects a list of color names, got {type(color_names)}")
        
        # 确保所有元素都是字符串
        color_names = [str(name) for name in color_names]
        
        return self.forward(color_names)


# -------------------------
# 3. 损失函数
# -------------------------
def binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary):
    """
    二分类损失函数，用于NCF模型的相似度分数
    
    Args:
        similarity_score_positive: 正样本相似度分数 (batch_size,)
        similarity_score_negative: 负样本相似度分数 (batch_size,)
        positive_labels_binary: 正样本标签 (batch_size,)
        negative_labels_binary: 负样本标签 (batch_size,)
    
    Returns:
        loss: 二分类损失
    """
    
    # 对于NCF模型，使用二分类损失
    # 正样本损失
    positive_loss = F.binary_cross_entropy(similarity_score_positive.squeeze(), positive_labels_binary.float())
    # 负样本损失
    negative_loss = F.binary_cross_entropy(similarity_score_negative.squeeze(), negative_labels_binary.float())
    
    # 总二分类损失
    loss = positive_loss + negative_loss
    return loss

def contrastive_loss_with_negatives(rgb_embeddings, name_embeddings, negative_embeddings, temperature=0.07):
    """
    对比学习损失函数，支持负样本
    
    Args:
        rgb_embeddings: RGB嵌入 (batch_size, emb_dim)
        name_embeddings: 名称嵌入 (batch_size, emb_dim)
        labels: 正样本标签 (batch_size,)
        negative_labels: 负样本标签 (batch_size,)
        name_encoder: 名称编码器（用于预训练Transformer）
        unique_terms: 唯一颜色名称列表（用于预训练Transformer）
        temperature: 温度参数，用于控制相似度分数的分布
    
    Returns:
        loss: 对比学习损失
    """
    batch_size = rgb_embeddings.size(0)
    
    # 计算正样本对的相似度
    positive_similarities = torch.sum(rgb_embeddings * name_embeddings, dim=1) / temperature
    
    # 计算负样本对的相似度
    negative_similarities = torch.sum(rgb_embeddings * negative_embeddings, dim=1) / temperature
    
    # 对比学习损失：最大化正样本相似度，最小化负样本相似度
    # 使用InfoNCE损失
    logits = torch.cat([positive_similarities.unsqueeze(1), negative_similarities.unsqueeze(1)], dim=1)
    targets = torch.zeros(batch_size, dtype=torch.long, device=rgb_embeddings.device)  # 正样本在第一个位置
    
    # 添加调试信息
    if torch.isnan(logits).any() or torch.isinf(logits).any():
        logger.info(f"警告: logits包含NaN或Inf值")
        logger.info(f"正样本相似度: {positive_similarities[:5]}")
        logger.info(f"负样本相似度: {negative_similarities[:5]}")
        logger.info(f"logits形状: {logits.shape}")
    
    loss = F.cross_entropy(logits, targets)
    
    # 检查损失值
    if torch.isnan(loss) or torch.isinf(loss):
        logger.info(f"警告: 对比学习损失为NaN或Inf: {loss}")
        # 返回一个小的非零损失作为fallback
        return torch.tensor(0.1, device=rgb_embeddings.device, requires_grad=True)
    
    return loss

def color_difference_loss(predicted_rgb, target_rgb):
    """
    颜色差异损失函数
    基于CIELAB颜色空间计算预测RGB与目标RGB的距离（可微分版本）
    
    Args:
        predicted_rgb: 预测的RGB值 [batch_size, 3] (0-1范围)
        target_rgb: 目标的RGB值 [batch_size, 3] (0-1范围)
    
    Returns:
        loss: 颜色差异损失
    """
    # 使用可微分的RGB到LAB转换
    pred_lab = rgb_to_lab_torch(predicted_rgb)  # [batch_size, 3]
    targ_lab = rgb_to_lab_torch(target_rgb)     # [batch_size, 3]
    
    # 计算CIELAB距离的平方根（可微分）
    lab_distance = torch.sqrt(torch.sum((pred_lab - targ_lab)**2, dim=1))  # [batch_size]
    
    # 返回平均损失
    return torch.mean(lab_distance)

# -------------------------
# 4. 数据集
# -------------------------
class ColorNameDatasetWithNegatives(Dataset):
    """
    支持负样本的颜色名称数据集
    为每个正样本生成对应的负样本，使用基于LAB距离的概率采样策略
    支持GPU并行计算优化
    """
    def __init__(self, rgb_values, encoded_labels, unique_terms, negative_ratio=1.0, 
                 use_lab_distance_sampling=False, sample_size=100, use_gpu=False, device=None):
        self.rgb_values = rgb_values
        self.encoded_labels = encoded_labels
        self.unique_terms = unique_terms
        self.negative_ratio = negative_ratio
        self.vocab_size = len(unique_terms)
        self.use_lab_distance_sampling = use_lab_distance_sampling
        self.sample_size = sample_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = device if device is not None else (torch.device('cuda') if self.use_gpu else torch.device('cpu'))
        
        self.rgb_values_ab = np.random.randint (0, 255, rgb_values.shape)
        self.rgb_values_ab = self.rgb_values_ab / 255
        # 为每个样本生成负样本标签
        logger.info(f"正在为 {len(rgb_values)} 个样本生成负样本标签...")
        if self.use_lab_distance_sampling:
            logger.info(f"使用LAB距离概率采样策略，采样数量: {self.sample_size}")
        if self.use_gpu:
            logger.info(f"使用GPU并行计算优化，设备: {self.device}")
        start_time = time.time()
        self.negative_labels = self._generate_negative_labels()
        end_time = time.time()
        logger.info(f"负样本标签生成完成，耗时: {end_time - start_time:.2f} 秒")
    
    def _generate_negative_labels(self):
        """为每个正样本生成负样本标签 - 基于LAB距离的概率采样"""
        if self.use_gpu and self.use_lab_distance_sampling:
            return self._generate_negative_labels_gpu()
        else:
            return self._generate_negative_labels_cpu()
    
    def _generate_negative_labels_gpu(self):
        """GPU并行计算版本的负样本生成"""
        logger.info("使用GPU并行计算生成负样本标签...")
        
        # 将数据转换为GPU张量
        rgb_tensor = torch.tensor(self.rgb_values[:, :3], dtype=torch.float32, device=self.device)
        labels_tensor = torch.tensor(self.encoded_labels, dtype=torch.long, device=self.device)
        
        # 批量计算所有RGB的LAB值
        logger.info("批量计算LAB值...")
        lab_tensor = rgb_to_lab_torch(rgb_tensor)  # [N, 3]
        
        n_samples = len(self.encoded_labels)
        negative_labels = []
        
        # 批量处理，避免内存溢出
        batch_size = min(100000, n_samples)  # 根据GPU内存调整批次大小
        
        for batch_start in range(0, n_samples, batch_size):
            batch_end = min(batch_start + batch_size, n_samples)
            batch_indices = torch.arange(batch_start, batch_end, device=self.device)
            
            # 当前批次的标签
            current_labels = labels_tensor[batch_indices]  # [batch_size]
            
            # 为每个样本随机选择候选样本
            candidate_indices = torch.randint(0, len(self.rgb_values), 
                                            (batch_end - batch_start, self.sample_size), 
                                            device=self.device)
            
            # 计算LAB距离矩阵
            current_lab = lab_tensor[batch_indices]  # [batch_size, 3]
            candidate_lab = lab_tensor[candidate_indices]  # [batch_size, sample_size, 3]
            
            # 计算欧几里得距离
            lab_distances = torch.norm(
                current_lab.unsqueeze(1) - candidate_lab, dim=2
            )  # [batch_size, sample_size]
            
            # 获取候选标签
            candidate_labels = labels_tensor[candidate_indices]  # [batch_size, sample_size]
            
            # 为每个样本选择负样本
            for i, (current_label, distances, candidates) in enumerate(zip(
                current_labels, lab_distances, candidate_labels
            )):
                # 排除当前样本和相同标签的样本
                valid_mask = (candidates != current_label) & (candidates != current_labels[i])
                
                if valid_mask.any():
                    valid_distances = distances[valid_mask]
                    valid_candidates = candidates[valid_mask]
                    
                    # 使用softmax将距离转换为概率
                    temperature = 10.0
                    probabilities = torch.softmax(valid_distances / temperature, dim=0)
                    
                    # 根据概率采样
                    negative_label = torch.multinomial(probabilities, 1).item()
                    negative_labels.append(valid_candidates[negative_label].item())
                else:
                    # 如果没有找到合适的候选样本，随机选择
                    while True:
                        negative_label = torch.randint(0, self.vocab_size, (1,), device=self.device).item()
                        if negative_label != current_label.item():
                            negative_labels.append(negative_label)
                            break
            
            # 清理GPU内存
            if batch_start % (batch_size * 5) == 0:  # 每5个批次清理一次
                torch.cuda.empty_cache()
        
        return np.array(negative_labels)
    
    def _generate_negative_labels_cpu(self):
        """CPU版本的负样本生成（原始实现）"""
        logger.info("使用CPU计算生成负样本标签...")
        
        n_samples = len(self.encoded_labels)
        negative_labels = []
        
        if self.use_lab_distance_sampling:
            # 预计算所有RGB的LAB值（避免重复计算）
            logger.info("预计算LAB值...")
            rgb_3d = self.rgb_values[:, :3]  # 只取RGB通道
            lab_values = np.zeros((len(rgb_3d), 3))
            
            for i, rgb in enumerate(rgb_3d):
                try:
                    lab_values[i] = rgb_to_lab(rgb.tolist())
                except:
                    lab_values[i] = [50.0, 0.0, 0.0]  # 默认LAB值
            
            logger.info("LAB值预计算完成，开始采样...")
            # 使用LAB距离概率采样策略
            for i in range(n_samples):
                current_label = self.encoded_labels[i]
                
                # 随机选择100个样本
                candidate_indices = np.random.choice(
                    len(self.rgb_values), 
                    size=min(self.sample_size, len(self.rgb_values)), 
                    replace=False
                )
                
                # 计算LAB距离
                lab_distances = []
                for idx in candidate_indices:
                    if idx != i:  # 排除当前样本
                        candidate_label = self.encoded_labels[idx]
                        
                        # 只考虑不同标签的样本
                        if candidate_label != current_label:
                            # 计算LAB距离
                            try:
                                current_lab = lab_values[i]
                                candidate_lab = lab_values[idx]
                                lab_distance = np.sqrt(np.sum((np.array(current_lab) - np.array(candidate_lab))**2))
                                lab_distances.append((candidate_label, lab_distance))
                            except:
                                # 如果转换失败，使用随机标签
                                lab_distances.append((candidate_label, 0.0))
                
                if lab_distances:
                    # 将LAB距离转换为概率（距离越大，概率越大）
                    labels, distances = zip(*lab_distances)
                    distances = np.array(distances)
                    
                    # 使用softmax将距离转换为概率
                    # 添加温度参数来控制分布的尖锐程度
                    temperature = 10.0  # 可以调整这个参数
                    probabilities = np.exp(distances / temperature)
                    probabilities = probabilities / np.sum(probabilities)
                    
                    # 根据概率采样
                    negative_label = np.random.choice(labels, p=probabilities)
                else:
                    # 如果没有找到合适的候选样本，随机选择
                    while True:
                        negative_label = np.random.randint(0, self.vocab_size)
                        if negative_label != current_label:
                            break
                
                negative_labels.append(negative_label)
        else:
            # 使用原来的随机采样策略
            negative_labels = np.random.randint(0, self.vocab_size, size=n_samples)
            
            # 处理冲突：确保负样本标签不等于正样本标签
            conflicts = negative_labels == self.encoded_labels
            if conflicts.any():
                conflict_indices = np.where(conflicts)[0]
                for idx in conflict_indices:
                    true_label = self.encoded_labels[idx]
                    while True:
                        new_label = np.random.randint(0, self.vocab_size)
                        if new_label != true_label:
                            negative_labels[idx] = new_label
                            break
        
        return np.array(negative_labels)
    
    def __len__(self):
        return len(self.rgb_values)
    
    def __getitem__(self, idx):
        # 获取正样本数据
        rgb = self.rgb_values[idx]
        rgb_ab = self.rgb_values_ab[idx]
        positive_label = self.encoded_labels[idx]
        negative_label = self.negative_labels[idx]
        feature_input = rgb
        feature_input_ab = rgb_ab
        
        return (
            torch.tensor(feature_input, dtype=torch.float32),  # 特征输入
            torch.tensor(positive_label, dtype=torch.long),   # 正样本标签
            torch.tensor(negative_label, dtype=torch.long),    # 负样本标签
            torch.tensor(feature_input_ab, dtype=torch.float32)    # 负样本标签
        )


# -------------------------
# 5. 训练函数
# -------------------------
def train_pretrained_transformer_model(rgb_values, encoded_labels, unique_terms,
                                      model_name='bert-base-uncased', num_epochs=30, batch_size=64, 
                                      emb_dim=64, learning_rate=1e-4, save_dir='models',
                                      local_model_path=None, use_ncf=True, use_rgb_generator=True,name_frequency_threshold=100):
    """
    训练基于预训练Transformer的颜色名称模型（支持负样本和NCF架构）
    
    Args:
        rgb_values: RGB值数组
        encoded_labels: 编码后的标签
        unique_terms: 唯一颜色名称列表
        model_name: 预训练模型名称
        num_epochs: 训练轮数
        batch_size: 批次大小
        emb_dim: 嵌入维度
        learning_rate: 学习率
        save_dir: 模型保存目录
        local_model_path: 本地模型路径
        use_ncf: 是否使用NCF架构
        name_frequency_threshold: 偶数epoch时使用的高频颜色名称阈值
    
    Returns:
        ncf_model: 训练好的NCF模型（如果use_ncf=True）
        rgb_encoder: 训练好的RGB编码器
        name_encoder: 训练好的预训练Transformer名称编码器
        rgb_generator: 训练好的RGB生成器（如果use_rgb_generator=True）
    """
    # 创建保存目录
    os.makedirs(save_dir, exist_ok=True)
    
    # 计算每个颜色名称的出现次数
    name_counts = Counter(encoded_labels)
    logger.info(f"颜色名称出现次数统计:")
    logger.info(f"  总颜色名称数量: {len(unique_terms)}")
    logger.info(f"  高频颜色名称阈值: {name_frequency_threshold}")
    
    # 找出高频颜色名称
    high_freq_indices = [idx for idx, count in name_counts.items() if count >= name_frequency_threshold]
    high_freq_names = [unique_terms[idx] for idx in high_freq_indices]
    logger.info(f"  高频颜色名称数量: {len(high_freq_indices)}")
    logger.info(f"  前5个高频颜色名称: {high_freq_names[:5]}")
    
    # 创建高频数据子集
    high_freq_mask = np.isin(encoded_labels, high_freq_indices)
    high_freq_rgb = rgb_values[high_freq_mask]
    high_freq_labels = encoded_labels[high_freq_mask]
    
    logger.info(f"  高频数据子集大小: {len(high_freq_rgb)}，占总数据的比例: {len(high_freq_rgb)/len(rgb_values):.2%}")
    
    # 划分训练集和验证集（全部数据）
    train_rgb, val_rgb, train_labels, val_labels = train_test_split(
        rgb_values, encoded_labels, test_size=0.2, random_state=42
    )
    
    # 划分高频数据的训练集和验证集
    high_freq_train_rgb, high_freq_val_rgb, high_freq_train_labels, high_freq_val_labels = train_test_split(
        high_freq_rgb, high_freq_labels, test_size=0.2, random_state=42
    )
    
    logger.info(f"全部数据 - 训练集: {len(train_rgb)}, 验证集: {len(val_rgb)}")
    logger.info(f"高频数据 - 训练集: {len(high_freq_train_rgb)}, 验证集: {len(high_freq_val_rgb)}")
    
    # # 创建数据集
    # train_dataset = ColorNameDatasetWithNegatives(train_rgb, train_labels, unique_terms)
    # val_dataset = ColorNameDatasetWithNegatives(val_rgb, val_labels, unique_terms)
    # high_freq_train_dataset = ColorNameDatasetWithNegatives(high_freq_train_rgb, high_freq_train_labels, unique_terms)
    # high_freq_val_dataset = ColorNameDatasetWithNegatives(high_freq_val_rgb, high_freq_val_labels, unique_terms)
    
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    # high_freq_train_loader = DataLoader(high_freq_train_dataset, batch_size=batch_size, shuffle=True)
    # high_freq_val_loader = DataLoader(high_freq_val_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    rgb_encoder = RGBEncoder(emb_dim=emb_dim, input_dim=rgb_values.shape[1]).to(device)
    name_encoder = NameEncoder(
        model_name=model_name,
        emb_dim=emb_dim,
        freeze_pretrained=use_freezed_weight,  # 冻结预训练参数
        use_pooling=False,  # 禁用注意力池化，使用CLS token或平均池化
        local_model_path=local_model_path
    ).to(device)
    
    # 创建NCF模型
    if use_ncf:
        logger.info(f"使用NCF架构")
        ncf_model = NCFModel(
            emb_dim=emb_dim
        ).to(device)
    else:
        ncf_model = None
        logger.info("不使用NCF架构")

    # 创建RGB生成器
    if use_rgb_generator:
        logger.info(f"使用RGB生成器，嵌入维度: {emb_dim}")
        rgb_generator = RGBGenerator(
            input_dim=emb_dim,
            hidden_dims=rgb_generator_hidden_dims,
            output_dim=3
        ).to(device)
    else:
        rgb_generator = None
        logger.info("不使用RGB生成器")

    # 分层学习率优化器
    optimizer_params = [
        {'params': rgb_encoder.parameters(), 'lr': learning_rate},
        {'params': name_encoder.pretrained_model.parameters(), 'lr': learning_rate * 0.1},
        {'params': name_encoder.output_projection.parameters(), 'lr': learning_rate}
    ]
    # 只有当注意力池化存在时才添加其参数
    if hasattr(name_encoder, 'attention_pooling') and name_encoder.attention_pooling is not None:
        optimizer_params.append({'params': name_encoder.attention_pooling.parameters(), 'lr': learning_rate})
    if use_ncf:
        # 添加NCF模型参数
        optimizer_params.append({'params': ncf_model.fusion_mlp.parameters(), 'lr': learning_rate})
    if use_rgb_generator:
        # 添加RGB生成器参数
        optimizer_params.append({'params': rgb_generator.parameters(), 'lr': learning_rate})
    
    optimizer = torch.optim.AdamW(optimizer_params)

    # 学习率调度器
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    train_losses = []
    val_losses = []
    train_contrastive_losses = []
    val_contrastive_losses = []
    train_cls_losses = []
    val_cls_losses = []
    train_rgb_generation_losses = []
    val_rgb_generation_losses = []
    train_color_embedding_losses = []
    val_color_embedding_losses = []
    best_val_loss = float('inf')
    
    logger.info(f"开始训练预训练Transformer模型（ NCF: {use_ncf}），共 {num_epochs} 轮...")
    logger.info(f"使用组合损失函数: {contrastive_loss_weight}*contrastive_loss + {binary_loss_weight}*classification_loss + {color_distance_weight}*rgb_generation_loss + {color_embedding_weight}*color_embedding_loss")
    
    # 预计算所有颜色名称的embedding以提高训练效率
    if isinstance(unique_terms, np.ndarray):
        unique_terms_list = unique_terms.tolist()
    else:
        unique_terms_list = list(unique_terms)

    for epoch in range(num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        # 创建数据集
        train_dataset = ColorNameDatasetWithNegatives(train_rgb, train_labels, unique_terms)
        val_dataset = ColorNameDatasetWithNegatives(val_rgb, val_labels, unique_terms)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        if use_high_freq_data:
            # 根据epoch奇偶性选择训练数据
            if epoch % 2 == 1:  # 奇数epoch：使用高频数据
                high_freq_train_dataset = ColorNameDatasetWithNegatives(high_freq_train_rgb, high_freq_train_labels, unique_terms)
                high_freq_val_dataset = ColorNameDatasetWithNegatives(high_freq_val_rgb, high_freq_val_labels, unique_terms)
                high_freq_train_loader = DataLoader(high_freq_train_dataset, batch_size=batch_size, shuffle=True)
                high_freq_val_loader = DataLoader(high_freq_val_dataset, batch_size=batch_size, shuffle=False)
                current_train_loader = high_freq_train_loader
                current_val_loader = high_freq_val_loader
                data_type = "高频数据"
            else:  # 偶数epoch：使用全部数据
                current_train_loader = train_loader
                current_val_loader = val_loader
                data_type = "全部数据"
        else:
            current_train_loader = train_loader
            current_val_loader = val_loader
            data_type = "全部数据"

        logger.info(f"=== ({data_type}) ===")
        # 训练阶段
        rgb_encoder.train()
        name_encoder.train()
        if use_ncf:
            ncf_model.train()
        if use_rgb_generator:
            rgb_generator.train()
        
        train_loss = 0
        train_contrastive_loss = 0
        train_cls_loss = 0
        train_rgb_generation_loss = 0
        train_color_embedding_loss = 0
        start_time = time.time()

        for batch in current_train_loader:
            rgb_batch, positive_labels, negative_labels, rgb_ab_batch = batch
            rgb_batch = rgb_batch.to(device)
            positive_labels = positive_labels.to(device)
            negative_labels = negative_labels.to(device)
            rgb_ab_batch = rgb_ab_batch.to(device)
            
            # 获取正样本的颜色名称
            batch_positive_names = [unique_terms_list[idx] for idx in positive_labels.cpu().numpy()]
            
            z_rgb = rgb_encoder(rgb_batch)
            z_names_positive = name_encoder(batch_positive_names)
            # 创建正样本标签（全为1）
            positive_labels_binary = torch.ones(len(rgb_batch), device=rgb_batch.device)

            if binary_loss_weight != 0 or contrastive_loss_weight != 0:
                # 获取负样本的名称
                batch_negative_names = [unique_terms_list[idx] for idx in negative_labels.cpu().numpy()]
                z_names_negative = name_encoder(batch_negative_names)
                # 正样本标签为1，负样本标签为0
                negative_labels_binary = torch.zeros(len(rgb_batch), device=rgb_batch.device)

            # 1. 对比学习损失 (contrastive_loss_with_negatives)
            if contrastive_loss_weight != 0:
                contrastive_loss = contrastive_loss_with_negatives(z_rgb, z_names_positive, z_names_negative)
            else:
                contrastive_loss = 0
            
            # 2. 二分类损失 (binary_classification_loss)
            if binary_loss_weight != 0 and use_ncf and use_negtive_sample:
                similarity_score_positive = ncf_model(z_rgb, z_names_positive)
                similarity_score_negative = ncf_model(z_rgb, z_names_negative)
                
                # 总二分类损失
                cls_loss = binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary)
                if use_ab_data_weight != 0:
                    z_rgb_ab = rgb_encoder(rgb_ab_batch)
                    # 正样本标签为1，负样本标签为0
                    negative_labels_binary_ab = torch.zeros(len(rgb_batch), device=rgb_batch.device)

                    similarity_score_negative_ab = ncf_model(z_rgb_ab, z_names_positive)
                    cls_loss_ab = binary_classification_loss(similarity_score_positive, similarity_score_negative_ab, positive_labels_binary, negative_labels_binary_ab)
                    cls_loss = (cls_loss + use_ab_data_weight*cls_loss_ab)/2
            else:
                cls_loss = 0

            if not use_ncf:
                # 这里将cls_loss中的ncf换为内积
                # 内积结果范围是[-1,1]，需要sigmoid映射到[0,1]以适配binary_cross_entropy
                similarity_score_positive = torch.sigmoid((z_rgb * z_names_positive).sum(dim=1))
                similarity_score_negative = torch.sigmoid((z_rgb * z_names_negative).sum(dim=1))
                cls_loss = binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary)

            if not use_negtive_sample:
                similarity_score_positive = ncf_model(z_rgb, z_names_positive)
                # cls_loss = F.binary_cross_entropy(similarity_score_positive.squeeze(), positive_labels_binary.float())
                # 使用MSE损失，让NCF输出接近1
                cls_loss = F.mse_loss(similarity_score_positive.squeeze(), positive_labels_binary.float())
                
            # 3. RGB生成损失 (MSE loss)
            if color_distance_weight != 0:
                # 从名称嵌入生成RGB值
                generated_rgb = rgb_generator(z_names_positive)  # [batch_size, 3]
                
                # 获取真实的RGB值（前3维）
                target_rgb = rgb_batch[:, :3]  # [batch_size, 3]
                if use_lab_dist:
                    # 使用可微分的RGB到LAB转换（保持梯度）
                    generated_lab = rgb_to_lab_torch(generated_rgb)
                    target_lab = rgb_to_lab_torch(target_rgb)
                    
                    # 更新变量名以保持一致性
                    generated_rgb = generated_lab
                    target_rgb = target_lab
                
                # 计算MSE损失
                mse_loss = nn.MSELoss()
                rgb_generation_loss = mse_loss(generated_rgb, target_rgb)
            else:
                rgb_generation_loss = 0
            
            if color_embedding_weight != 0:
                # 颜色embedding损失
                # 获取rgb_generator倒数第三层的64维输出
                color_embedding = rgb_generator.mlp[-3](z_names_positive)  # 应该是64维
                # 使用余弦相似度损失计算rgbencoder与rgbgenerator生成的颜色embedding的损失
                color_embedding_loss = 1 - F.cosine_similarity(z_rgb, color_embedding, dim=1).mean()
            else:
                color_embedding_loss = 0

            # 组合损失
            loss = contrastive_loss_weight*contrastive_loss + binary_loss_weight*cls_loss + color_distance_weight*rgb_generation_loss + color_embedding_weight*color_embedding_loss
            
            # 记录各项损失
            train_contrastive_loss += contrastive_loss.item() if hasattr(contrastive_loss, 'item') else contrastive_loss
            train_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
            train_rgb_generation_loss += rgb_generation_loss.item() if hasattr(rgb_generation_loss, 'item') else rgb_generation_loss
            train_color_embedding_loss += color_embedding_loss.item() if hasattr(color_embedding_loss, 'item') else color_embedding_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_cls_loss = train_cls_loss / len(train_loader)
        avg_train_contrastive_loss = train_contrastive_loss / len(train_loader)
        avg_train_rgb_generation_loss = train_rgb_generation_loss / len(train_loader)
        avg_train_color_embedding_loss = train_color_embedding_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_cls_losses.append(avg_train_cls_loss)
        train_contrastive_losses.append(avg_train_contrastive_loss)
        train_rgb_generation_losses.append(avg_train_rgb_generation_loss)
        train_color_embedding_losses.append(avg_train_color_embedding_loss)

        # 验证阶段
        rgb_encoder.eval()
        name_encoder.eval()
        if use_ncf:
            ncf_model.eval()
        if use_rgb_generator:
            rgb_generator.eval()
            
        val_loss = 0
        val_cls_loss = 0
        val_contrastive_loss = 0
        val_rgb_generation_loss = 0
        val_color_embedding_loss = 0
        with torch.no_grad():
            for batch in current_val_loader:
                rgb_batch, positive_labels, negative_labels, rgb_ab_batch = batch
                rgb_batch = rgb_batch.to(device)
                positive_labels = positive_labels.to(device)
                negative_labels = negative_labels.to(device)
                rgb_ab_batch = rgb_ab_batch.to(device)
                
                # 获取正样本的颜色名称
                batch_positive_names = [unique_terms_list[idx] for idx in positive_labels.cpu().numpy()]
                
                z_rgb = rgb_encoder(rgb_batch)
                z_names_positive = name_encoder(batch_positive_names)
                # 创建正样本标签（全为1）
                positive_labels_binary = torch.ones(len(rgb_batch), device=rgb_batch.device)

                if binary_loss_weight != 0 or contrastive_loss_weight != 0:
                    # 获取负样本的名称
                    batch_negative_names = [unique_terms_list[idx] for idx in negative_labels.cpu().numpy()]
                    z_names_negative = name_encoder(batch_negative_names)
                    # 正样本标签为1，负样本标签为0
                    negative_labels_binary = torch.zeros(len(rgb_batch), device=rgb_batch.device)

                 # 1. 对比学习损失 (contrastive_loss_with_negatives)
                if contrastive_loss_weight != 0:
                    contrastive_loss = contrastive_loss_with_negatives(z_rgb, z_names_positive, z_names_negative)
                else:
                    contrastive_loss = 0
                
                # 2. 二分类损失 (binary_classification_loss)
                if binary_loss_weight != 0 and use_ncf and use_negtive_sample:
                    similarity_score_positive = ncf_model(z_rgb, z_names_positive)
                    similarity_score_negative = ncf_model(z_rgb, z_names_negative)
                    
                    # 总二分类损失
                    cls_loss = binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary)
                    if use_ab_data_weight != 0:
                        z_rgb_ab = rgb_encoder(rgb_ab_batch)
                        # 正样本标签为1，负样本标签为0
                        negative_labels_binary_ab = torch.zeros(len(rgb_batch), device=rgb_batch.device)

                        similarity_score_negative_ab = ncf_model(z_rgb_ab, z_names_positive)
                        cls_loss_ab = binary_classification_loss(similarity_score_positive, similarity_score_negative_ab, positive_labels_binary, negative_labels_binary_ab)
                        cls_loss = (cls_loss + use_ab_data_weight*cls_loss_ab)/2
                else:
                    cls_loss = 0

                if not use_ncf:
                    # 这里将cls_loss中的ncf换为内积
                    # 内积结果范围是[-1,1]，需要sigmoid映射到[0,1]以适配binary_cross_entropy
                    similarity_score_positive = torch.sigmoid((z_rgb * z_names_positive).sum(dim=1))
                    similarity_score_negative = torch.sigmoid((z_rgb * z_names_negative).sum(dim=1))
                    cls_loss = binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary)

                if not use_negtive_sample:
                    similarity_score_positive = ncf_model(z_rgb, z_names_positive)
                    # cls_loss = F.binary_cross_entropy(similarity_score_positive.squeeze(), positive_labels_binary.float())
                    # 使用MSE损失，让NCF输出接近1
                    cls_loss = F.mse_loss(similarity_score_positive.squeeze(), positive_labels_binary.float())
                else:
                    cls_loss = binary_classification_loss(similarity_score_positive, similarity_score_negative, positive_labels_binary, negative_labels_binary)

                # 3. RGB生成损失 (MSE loss)
                if color_distance_weight != 0:
                    # 从名称嵌入生成RGB值
                    generated_rgb = rgb_generator(z_names_positive)  # [batch_size, 3]
                    
                    # 获取真实的RGB值（前3维）
                    target_rgb = rgb_batch[:, :3]  # [batch_size, 3]
                    if use_lab_dist:
                        # 使用可微分的RGB到LAB转换（保持梯度）
                        generated_lab = rgb_to_lab_torch(generated_rgb)
                        target_lab = rgb_to_lab_torch(target_rgb)
                        
                        # 更新变量名以保持一致性
                        generated_rgb = generated_lab
                        target_rgb = target_lab

                    # 计算MSE损失
                    mse_loss = nn.MSELoss()
                    rgb_generation_loss = mse_loss(generated_rgb, target_rgb)
                else:
                    rgb_generation_loss = 0
                
                if color_embedding_weight != 0:
                    # 颜色embedding损失
                    # 获取rgb_generator倒数第三层的64维输出
                    color_embedding = rgb_generator.mlp[-3](z_names_positive)  # 应该是64维
                    # 使用余弦相似度损失计算rgbencoder与rgbgenerator生成的颜色embedding的损失
                    color_embedding_loss = 1 - F.cosine_similarity(z_rgb, color_embedding, dim=1).mean()
                else:
                    color_embedding_loss = 0

                # 组合损失
                loss = contrastive_loss_weight*contrastive_loss + binary_loss_weight*cls_loss + color_distance_weight*rgb_generation_loss + color_embedding_weight*color_embedding_loss
                
                # 记录各项损失
                val_cls_loss += cls_loss.item() if hasattr(cls_loss, 'item') else cls_loss
                val_contrastive_loss += contrastive_loss.item() if hasattr(contrastive_loss, 'item') else contrastive_loss
                val_rgb_generation_loss += rgb_generation_loss.item() if hasattr(rgb_generation_loss, 'item') else rgb_generation_loss
                val_color_embedding_loss += color_embedding_loss.item() if hasattr(color_embedding_loss, 'item') else color_embedding_loss
                val_loss += loss.item()
                    
        avg_val_loss = val_loss / len(val_loader)
        avg_val_cls_loss = val_cls_loss / len(val_loader)
        avg_val_contrastive_loss = val_contrastive_loss / len(val_loader)
        avg_val_rgb_generation_loss = val_rgb_generation_loss / len(val_loader)
        avg_val_color_embedding_loss = val_color_embedding_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_cls_losses.append(avg_val_cls_loss)
        val_contrastive_losses.append(avg_val_contrastive_loss)
        val_rgb_generation_losses.append(avg_val_rgb_generation_loss)
        val_color_embedding_losses.append(avg_val_color_embedding_loss)
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"  训练损失: {avg_train_loss:.4f} (contrastive loss: {avg_train_contrastive_loss:.4f}, binary loss: {avg_train_cls_loss:.4f}, MSE loss: {avg_train_rgb_generation_loss:.4f}, Cosine loss: {avg_train_color_embedding_loss:.4f})")
        logger.info(f"  验证损失: {avg_val_loss:.4f} (contrastive loss: {avg_val_contrastive_loss:.4f}, binary loss: {avg_val_cls_loss:.4f}, MSE loss: {avg_val_rgb_generation_loss:.4f}, Cosine loss: {avg_val_color_embedding_loss:.4f})")
        logger.info(f"  训练时间: {(time.time() - start_time)/3600:.2f} 小时")
        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_name = f'model_best{suffix_str}.pt'
            
            # 保存模型
            save_dict = {
                'rgb_encoder': rgb_encoder.state_dict(),
                'name_encoder': name_encoder.state_dict(),
                'epoch': epoch,
                'val_loss': avg_val_loss,
                'use_ncf': use_ncf,
                'use_rgb_generator': use_rgb_generator,
                'model_name': model_name
            }
            
            if use_ncf:
                save_dict['ncf_model'] = ncf_model.state_dict()
            
            if use_rgb_generator:
                save_dict['rgb_generator'] = rgb_generator.state_dict()
            
            torch.save(save_dict, os.path.join(save_dir, model_name))
            
            logger.info(f"  保存最佳预训练Transformer模型 (验证损失: {avg_val_loss:.4f})")
    
    # 绘制损失曲线
    plt.figure(figsize=(25, 5))
    
    # 总损失
    plt.subplot(1, 5, 1)
    plt.plot(train_losses, label='训练损失', color='blue')
    plt.plot(val_losses, label='验证损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    title = '总损失'
    plt.title(title)
    plt.legend()
    plt.grid(True)
    
    # 对比学习损失
    plt.subplot(1, 5, 2)
    plt.plot(train_contrastive_losses, label='训练对比损失', color='blue')
    plt.plot(val_contrastive_losses, label='验证对比损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Contrastive Loss')
    plt.title('Contrastive Loss')
    plt.legend()
    plt.grid(True)

    # 分类损失
    plt.subplot(1, 5, 3)
    plt.plot(train_cls_losses, label='训练分类损失', color='blue')
    plt.plot(val_cls_losses, label='验证分类损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Binary Loss')
    plt.title('Binary Loss')
    plt.legend()
    plt.grid(True)
    
    # RGB生成损失
    plt.subplot(1, 5, 4)
    plt.plot(train_rgb_generation_losses, label='训练RGB生成损失', color='blue')
    plt.plot(val_rgb_generation_losses, label='验证RGB生成损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('RGB Generation Loss')
    plt.legend()
    plt.grid(True)

    # 颜色embedding损失
    plt.subplot(1, 5, 5)
    plt.plot(train_color_embedding_losses, label='训练颜色embedding损失', color='blue')
    plt.plot(val_color_embedding_losses, label='验证颜色embedding损失', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Cosine Loss')
    plt.title('Color Embedding Loss')
    plt.legend()
    plt.grid(True)

    loss_plot_name = f'plot_loss_curve{suffix_str}.png'
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, loss_plot_name), dpi=300, bbox_inches='tight')
    plt.close()

    # 加载保存的最优模型到 ncf_model, rgb_encoder, name_encoder, rgb_generator
    best_model_path = os.path.join(save_dir, f'model_best{suffix_str}.pt')
    if not os.path.exists(best_model_path):
        logger.info(f"未找到模型文件: {best_model_path}")
    else:
        logger.info(f"加载最佳模型权重: {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location='cpu')
        if 'ncf_model' in checkpoint and ncf_model is not None:
            ncf_model.load_state_dict(checkpoint['ncf_model'])
        if 'rgb_encoder' in checkpoint and rgb_encoder is not None:
            rgb_encoder.load_state_dict(checkpoint['rgb_encoder'])
        if 'name_encoder' in checkpoint and name_encoder is not None:
            name_encoder.load_state_dict(checkpoint['name_encoder'])
        if 'rgb_generator' in checkpoint and rgb_generator is not None:
            rgb_generator.load_state_dict(checkpoint['rgb_generator'])
        logger.info("模型加载完毕。")
    
    return ncf_model, rgb_encoder, name_encoder, rgb_generator

# -------------------------
# 6. 推理函数
# -------------------------
def recommend_color_names(ncf_model, rgb_encoder, name_encoder, rgb_input, unique_terms, top_k=5):
    """
    使用NCF模型为给定的RGB值推荐颜色名称
    
    Args:
        ncf_model: NCF模型
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        rgb_input: RGB输入 (0-1范围)
        unique_terms: 颜色名称列表
        top_k: 推荐数量
    
    Returns:
        names: 推荐的颜色名称
        scores: 对应的分数
    """
    device = next(ncf_model.parameters()).device
    
    logger.info("RGB输入:", rgb_input)
    # 将RGB转换为16维特征
    if use_16d_feature:
        feature_16d = rgb_to_16d_feature(rgb_input)
        feature_tensor = torch.tensor(feature_16d, dtype=torch.float32).to(device)
    else:
        feature_tensor = torch.tensor(rgb_input, dtype=torch.float32).to(device)
    
    ncf_model.eval()
    rgb_encoder.eval()
    name_encoder.eval()
    
    with torch.no_grad():
        try:
            # 使用RGB编码器获取RGB embedding
            z_rgb = rgb_encoder(feature_tensor.unsqueeze(0))
            
            # 获取所有颜色名称的相似度分数
            all_scores = []
            for i, name in enumerate(unique_terms):
                # 使用名称编码器获取名称embedding
                z_name = name_encoder([name])
                
                # 使用NCF模型计算相似度分数
                similarity_score = ncf_model(z_rgb, z_name)
                all_scores.append(similarity_score.item())
                if i%10000 == 0:
                    logger.info(f"计算颜色名称相似度分数: {i}/{len(unique_terms)}")
            # 转换为numpy数组并排序
            all_scores = np.array(all_scores)
            top_indices = np.argsort(all_scores)[::-1][:top_k]
            
            names = [unique_terms[i] if isinstance(unique_terms, np.ndarray) else unique_terms[i] for i in top_indices]
            scores = all_scores[top_indices]
            
            return names, scores
            
        except Exception as e:
            logger.info(f"⚠️ NCF模型推理失败: {e}")
            return [], []

def recommend_rgb_colors(ncf_model, rgb_encoder, name_encoder, color_name, rgb_database, top_k=5):
    """
    使用NCF模型为给定的颜色名称推荐RGB值
    
    Args:
        ncf_model: NCF模型
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        color_name: 颜色名称（可以是未见过的）
        rgb_database: 16维特征数据库
        top_k: 推荐数量
    
    Returns:
        rgb_values: 推荐的RGB值
        scores: 对应的分数
    """
    device = next(ncf_model.parameters()).device
    logger.info("颜色名称:", color_name)

    ncf_model.eval()
    rgb_encoder.eval()
    name_encoder.eval()
    
    with torch.no_grad():
        try:
            # 使用名称编码器获取颜色名称的embedding
            z_name = name_encoder([color_name])
            
            # 根据use_16d_feature设置处理RGB特征
            rgb_features = []
            for rgb in rgb_database:
                if use_16d_feature:
                    # 使用16维特征
                    if len(rgb) == 3:
                        feature = rgb_to_16d_feature(rgb)
                    elif len(rgb) == 16:
                        feature = rgb
                    else:
                        logger.info(f"警告: 意外的RGB维度 {len(rgb)}，使用简单填充")
                        feature = list(rgb[:3]) + [0.0] * 13
                else:
                    # 使用3维RGB特征
                    if len(rgb) >= 3:
                        feature = rgb[:3]  # 只取前3维
                    else:
                        logger.info(f"警告: RGB维度不足 {len(rgb)}，使用零填充")
                        feature = list(rgb) + [0.0] * (3 - len(rgb))
                
                rgb_features.append(feature)
            
            # 计算每个RGB值与颜色名称的相似度分数
            all_scores = []
            for i, rgb_feature in enumerate(rgb_features):
                # 使用RGB编码器获取RGB embedding
                rgb_tensor = torch.tensor([rgb_feature], dtype=torch.float32).to(device)
                z_rgb = rgb_encoder(rgb_tensor)
                
                # 使用NCF模型计算相似度分数
                similarity_score = ncf_model(z_rgb, z_name)
                all_scores.append(similarity_score.item())
                if i%10000 == 0:
                    logger.info(f"计算颜色名称相似度分数: {i}/{len(rgb_features)}")
            
            # 转换为numpy数组并排序
            all_scores = np.array(all_scores)
            top_indices = np.argsort(all_scores)[::-1][:top_k]
            
            # 提取前3维作为RGB值
            rgb_values = rgb_database[top_indices][:, :3] if len(rgb_database.shape) > 1 else [rgb_database[top_indices][:3]]
            scores = all_scores[top_indices]
            
            return rgb_values, scores
            
        except Exception as e:
            logger.info(f"⚠️ NCF模型推理失败: {e}")
            return [], []


def load_xkcd_color_mapping():
    """加载xkcd颜色映射，构建term到LAB颜色的字典"""
    color_mapping = {}
    
    with open('xkcd-rgb-term.txt', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                # 提取颜色名称和十六进制值
                parts = line.split('\t')
                if len(parts) >= 2:
                    term = parts[0].strip()
                    hex_color = parts[1].strip()
                    
                    try:
                        # 转换十六进制为RGB
                        r, g, b = hex_to_rgb(hex_color)
                        
                        # 转换为0-1范围的RGB
                        rgb_01 = rgb_255_to_01(r, g, b)
                        
                        # 转换为LAB颜色空间
                        lab_color = rgb_to_lab(rgb_01)
                        
                        # 存储到字典中
                        color_mapping[term] = {
                            'hex': hex_color,
                            'rgb': (r, g, b),
                            'lab': lab_color
                        }
                        
                    except Exception as e:
                        logger.info(f"处理颜色 '{term}' 时出错: {e}")
    
    return color_mapping


# -------------------------
# 6. 颜色可视化函数
# -------------------------
def render_color_names(rgb_values, encoded_labels, unique_terms, unique_colors):
    """
    从rgb_values中提取所有name为green、blue、teal的颜色，分别将这些name对应的unique颜色绘制到3D的RGB cube里，
    每个颜色为一个小球，球的颜色为该颜色，并筛选出3个有代表性的颜色，将其对应小球的半径放大
    
    Args:
        rgb_values: 归一化的RGB值数组 (N, 3)，值范围0-1
        encoded_labels: 编码后的标签数组 (N,)，每个元素是对应颜色的名称编码
        unique_terms: 所有唯一颜色名称的数组
        unique_colors: 所有唯一RGB值的数组 (M, 3)
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    
    # 目标颜色名称
    target_names = ['green', 'blue', 'teal']
    
    # 创建标签编码器的反向映射
    label_to_term = {i: term for i, term in enumerate(unique_terms)}
    
    # 筛选出目标颜色的数据（只使用unique colors）
    target_data = {}
    for target_name in target_names:
        # 找到包含目标名称的所有标签索引
        target_indices = []
        for i, term in enumerate(unique_terms):
            if target_name.lower() == term.lower():
                target_indices.append(i)
        
        if target_indices:
            # 找到所有属于这些标签的RGB值
            mask = np.isin(encoded_labels, target_indices)
            target_rgb = rgb_values[mask]
            
            # 只保留唯一的RGB值
            target_rgb_unique = np.unique(target_rgb, axis=0)
            # 随机选择1000个unique颜色（如果数量大于1000）
            if len(target_rgb_unique) > 1000:
                idxs = np.random.choice(len(target_rgb_unique), 1000, replace=False)
                target_rgb_unique = target_rgb_unique[idxs]

            target_data[target_name] = target_rgb_unique
            logger.info(f"找到 {target_name} 相关颜色: {len(target_rgb)} 个，去重后: {len(target_rgb_unique)} 个")
        else:
            logger.warning(f"未找到 {target_name} 相关颜色")
            target_data[target_name] = np.array([])
    
    # 创建3D图形
    fig = plt.figure(figsize=(18, 5))
    
    # 为每个目标颜色创建子图
    for idx, (target_name, target_rgb) in enumerate(target_data.items()):
        if len(target_rgb) == 0:
            continue
            
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        
        # 绘制所有颜色的小球
        for i, rgb in enumerate(target_rgb):
            # 将归一化的RGB值转换回0-255范围用于显示
            display_rgb = rgb * 255
            color = [display_rgb[0]/255, display_rgb[1]/255, display_rgb[2]/255]
            
            # 绘制小球
            ax.scatter(rgb[0], rgb[1], rgb[2], c=[color], s=20, alpha=0.7)
        
        # 筛选出3个有代表性的颜色（使用K-means聚类确保差异）
        # if len(target_rgb) >= 3:
        #     from sklearn.cluster import KMeans
            
        #     # 使用K-means聚类，选择3个聚类中心作为代表性颜色
        #     kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
        #     cluster_labels = kmeans.fit_predict(target_rgb)
        #     representative_colors = kmeans.cluster_centers_
            
        #     # 为每个聚类中心找到最接近的实际颜色点
        #     representative_indices = []
        #     for center in representative_colors:
        #         distances = np.linalg.norm(target_rgb - center, axis=1)
        #         closest_idx = np.argmin(distances)
        #         representative_indices.append(closest_idx)
            
        #     # 计算聚类中心之间的距离，确保有足够的差异
        #     center_distances = []
        #     for i in range(len(representative_colors)):
        #         for j in range(i+1, len(representative_colors)):
        #             dist = np.linalg.norm(representative_colors[i] - representative_colors[j])
        #             center_distances.append(dist)
            
        #     avg_distance = np.mean(center_distances)
        #     min_distance = np.min(center_distances)
            
        #     logger.info(f"{target_name} 聚类结果:")
        #     logger.info(f"  聚类中心平均距离: {avg_distance:.3f}")
        #     logger.info(f"  聚类中心最小距离: {min_distance:.3f}")
            
        #     for i, (center, idx) in enumerate(zip(representative_colors, representative_indices)):
        #         actual_rgb = target_rgb[idx]
        #         logger.info(f"  聚类 {i+1}: 中心 {center}, 实际颜色 {actual_rgb}")
            
        #     # 如果聚类中心距离太近，尝试增加聚类数或使用其他方法
        #     if min_distance < 0.1:  # 如果最小距离小于0.1，认为差异不够
        #         logger.warning(f"{target_name} 聚类中心距离过近，尝试使用更分散的采样方法")
        #         # 使用最大最小距离采样
        #         representative_indices = []
        #         # 选择第一个点（随机或基于某种启发式）
        #         first_idx = np.random.randint(0, len(target_rgb))
        #         representative_indices.append(first_idx)
                
        #         # 选择距离第一个点最远的点
        #         distances = np.linalg.norm(target_rgb - target_rgb[first_idx], axis=1)
        #         second_idx = np.argmax(distances)
        #         representative_indices.append(second_idx)
                
        #         # 选择距离前两个点都最远的点
        #         min_distances = np.minimum(
        #             np.linalg.norm(target_rgb - target_rgb[first_idx], axis=1),
        #             np.linalg.norm(target_rgb - target_rgb[second_idx], axis=1)
        #         )
        #         third_idx = np.argmax(min_distances)
        #         representative_indices.append(third_idx)
                
        #         logger.info(f"{target_name} 使用最大最小距离采样方法")
            
        #     # 放大这3个代表性颜色的小球
        #     for rep_idx in representative_indices:
        #         rgb = target_rgb[rep_idx]
        #         display_rgb = rgb * 255
        #         color = [display_rgb[0]/255, display_rgb[1]/255, display_rgb[2]/255]
        #         ax.scatter(rgb[0], rgb[1], rgb[2], c=[color], s=100, alpha=0.9, edgecolors='black', linewidth=1)
        
        # 设置坐标轴
        ax.set_xlabel('Red')
        ax.set_ylabel('Green')
        ax.set_zlabel('Blue')
        ax.set_title(f'{target_name.capitalize()} Colors in RGB Cube')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(0, 1)
        
        # 设置视角
        ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig('color_names_3d_visualization.png', dpi=300, bbox_inches='tight')
    logger.info("3D颜色可视化完成，图片已保存为 color_names_3d_visualization.png")
    # plt.show()
    # 分别保存三个子图
    for idx, (target_name, target_rgb) in enumerate(target_data.items()):
        if len(target_rgb) == 0:
            continue
            
        # 为每个子图创建单独的图形
        fig_single = plt.figure(figsize=(8, 6))
        ax_single = fig_single.add_subplot(111, projection='3d')
        
        # 绘制所有颜色的小球
        for i, rgb in enumerate(target_rgb):
            # 将归一化的RGB值转换回0-255范围用于显示
            display_rgb = rgb * 255
            color = [display_rgb[0]/255, display_rgb[1]/255, display_rgb[2]/255]
            
            # 绘制小球
            ax_single.scatter(rgb[0], rgb[1], rgb[2], c=[color], s=20, alpha=1)
        
        # 设置坐标轴
        # ax_single.set_xlabel('Red', labelpad=10)
        # ax_single.set_ylabel('Green', labelpad=10)
        # ax_single.set_zlabel('Blue', labelpad=10)
        # ax_single.set_title(f'{target_name.capitalize()} Colors in RGB Cube')
        ax_single.set_xlim(0, 1)
        ax_single.set_ylim(0, 1)
        ax_single.set_zlim(0, 1)
        
        # 设置视角
        ax_single.view_init(elev=20, azim=45)
        
        # 保存单个子图
        plt.tight_layout()
        # 调整子图布局，确保坐标轴标题完全可见
        # plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(f'color_names_3d_visualization_subplot_{idx+1}.png', dpi=300, bbox_inches='tight')
        plt.close(fig_single)  # 关闭图形以释放内存
        logger.info(f"子图{idx+1} ({target_name}) 已保存为 color_names_3d_visualization_subplot_{idx+1}.png")
    

# -------------------------
# 7. 主训练流程
# -------------------------
def main():
    logger.info("=== 颜色名称模型训练 ===")
    
    # 1. 数据预处理
    rgb_values, encoded_labels, label_encoder, unique_terms, unique_colors = preprocess_data(
        csv_path='responses_cleaned_all.csv', min_count=10
    )
    # render_color_names(rgb_values, encoded_labels, unique_terms, unique_colors)
    # return
    # c3_terms = ["green","blue","purple","red","pink","yellow","orange","brown","teal","lightblue","grey","limegreen","magenta","lightgreen","brightgreen","skyblue","cyan","turquoise","darkblue","darkgreen","aqua","olive","navyblue","lavender","fuchsia","black","royalblue","violet","hotpink","tan","forestgreen","lightpurple","neongreen","yellowgreen","maroon","darkpurple","salmon","peach","beige","lime","seafoamgreen","mustard","brightblue","lilac","seagreen","palegreen","bluegreen","mint","lightbrown","mauve","darkred","greyblue","burntorange","darkpink","indigo","periwinkle","bluegrey","lightpink","aquamarine","gold","brightpurple","grassgreen","redorange","bluepurple","greygreen","kellygreen","puke","rose","darkteal","babyblue","paleblue","greenyellow","brickred","lightgrey","darkgrey","white","brightpink","chartreuse","purpleblue","royalpurple","burgundy","goldenrod","darkbrown","lightorange","darkorange","redbrown","paleyellow","plum","offwhite","pinkpurple","darkyellow","lightyellow","mustardyellow","brightred","peagreen","khaki","orangered","crimson","deepblue","springgreen","cream","palepink","yelloworange","deeppurple","pinkred","pastelgreen","sand","rust","lightred","taupe","armygreen","robinseggblue","huntergreen","greenblue","lightteal","cerulean","flesh","orangebrown","slateblue","slate","coral","blueviolet","ochre","leafgreen","electricblue","seablue","midnightblue","steelblue","brick","palepurple","mediumblue","burntsienna","darkmagenta","eggplant","sage","darkturquoise","puce","bloodred","neonpurple","mossgreen","terracotta","oceanblue","yellowbrown","brightyellow","dustyrose","applegreen","neonpink","skin","cornflowerblue","lightturquoise","wine","deepred","azure"]
    # # 从rgb_values和encoded_labels中，随机选择10000个颜色名属于c3_terms的样本
    # # 先获取c3_terms对应的label编码
    # c3_term_set = set([t.lower().replace(" ", "") for t in c3_terms])
    # # label_encoder.classes_ 里的term可能有空格，统一处理
    # c3_label_indices = []
    # for idx, term in enumerate(label_encoder.classes_):
    #     norm_term = term.lower().replace(" ", "")
    #     if norm_term in c3_term_set:
    #         c3_label_indices.append(idx)
    # c3_label_indices = set(c3_label_indices)
    # # 找到属于c3_terms的样本索引
    # c3_sample_indices = [i for i, label in enumerate(encoded_labels) if label in c3_label_indices]
    # # 如果数量不足10000，全部取出，否则随机采样10000个
    # if len(c3_sample_indices) < 10000:
    #     logger.info(f"c3_terms样本数不足10000，仅有{len(c3_sample_indices)}个，全部选用")
    #     selected_indices = c3_sample_indices
    # else:
    #     selected_indices = np.random.choice(c3_sample_indices, 10000, replace=False)
    
    # 从rgb_values中随机选择10000个样本
    # total_samples = len(rgb_values)
    # if total_samples <= 10000:
    #     logger.info(f"样本总数不足10000，仅有{total_samples}个，全部选用")
    #     selected_indices = np.arange(total_samples)
    # else:
    #     selected_indices = np.random.choice(total_samples, 10000, replace=False)
    # # 选出对应的rgb_values和encoded_labels
    # rgb_values_c3 = rgb_values[selected_indices]
    # encoded_labels_c3 = encoded_labels[selected_indices]
    # # 将选中的颜色及对应名称输出到csv
    # import csv
    # output_path = "selected_c3_colors-12.csv"
    # with open(output_path, "w", newline='', encoding="utf-8") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["r", "g", "b", "term"])
    #     for rgb, label in zip(rgb_values_c3, encoded_labels_c3):
    #         # rgb可能是16维，也可能是3维，这里只取前3维
    #         r, g, b = rgb[0], rgb[1], rgb[2]
    #         term = label_encoder.inverse_transform([label])[0]
    #         writer.writerow([r, g, b, term])
    # logger.info(f"已将{len(rgb_values_c3)}个c3_terms样本输出到 {output_path}")
    
    term_color_mapping = load_xkcd_color_mapping()
    logger.info(f"term_color_mapping including {len(term_color_mapping)} terms")

    # 将每个rgb_value转换为hsl、lab、hcl、cmyk，并将所有格式拼接为16维特征，替换原有rgb_values
    # 生成16维新特征
    if use_16d_feature:
        new_features = []
        for rgb in rgb_values:
            feature = rgb_to_16d_feature(rgb)
            new_features.append(feature)
        rgb_values = np.array(new_features, dtype=np.float32)

    logger.info(f"first 5 samples of rgb_values: {rgb_values[:5]}")
    # 保存预处理结果
    os.makedirs('models', exist_ok=True)
    with open('models/preprocessed_data.pkl', 'wb') as f:
        pickle.dump({
            'rgb_values': rgb_values,
            'encoded_labels': encoded_labels,
            'label_encoder': label_encoder,
            'unique_terms': unique_terms,
            'unique_colors': unique_colors
        }, f)

    # 划分训练集和测试集
    used_test_size = 10100 / len(rgb_values)
    logger.info(f"used_test_size: {used_test_size}")
    # 先全部分配给训练集，再手动划分10000个测试样本
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        rgb_values, encoded_labels, test_size=used_test_size, random_state=42
    )
    if len(X_test_full) > 10000:
        X_test = X_test_full[:10000]
        y_test = y_test_full[:10000]
        X_train = np.concatenate([X_train_full, X_test_full[10000:]], axis=0)
        y_train = np.concatenate([y_train_full, y_test_full[10000:]], axis=0)
    else:
        X_test = X_test_full
        y_test = y_test_full
        X_train = X_train_full
        y_train = y_train_full

    logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")

    # 统计y_train中出现的颜色名称及其出现次数
    name_count_dict = {}
    for idx in y_train:
        name = unique_terms[idx]
        if name in name_count_dict:
            name_count_dict[name] += 1
        else:
            name_count_dict[name] = 1
    logger.info(f"训练集中出现的颜色名称数量: {len(name_count_dict)}")
    # 打印出现次数最多的前20个颜色名称及其出现次数
    top20 = sorted(name_count_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("出现次数最多的前10个颜色名称及其出现次数：")
    for name, count in top20:
        logger.info(f"  {name}: {count}")
    logger.info('-'*30)
    color_database = np.unique(X_train, axis=0)
    # 只保留y_train中出现过的颜色名称，unique_terms为这些唯一颜色名的列表
    name_database = list(set([unique_terms[idx] for idx in y_train]))
    logger.info(f"训练集样本数: {len(X_train)}， 包含 {len(color_database)} 个唯一RGB值， 包含 {len(name_database)} 个唯一颜色名称")
    logger.info('*'*60)
    name_count_dict = {}
    for idx in y_test:
        name = unique_terms[idx]
        if name in name_count_dict:
            name_count_dict[name] += 1
        else:
            name_count_dict[name] = 1
    logger.info(f"测试集中出现的颜色名称数量: {len(name_count_dict)}")
    # 打印出现次数最多的前20个颜色名称及其出现次数
    top20 = sorted(name_count_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("出现次数最多的前10个颜色名称及其出现次数：")
    for name, count in top20:
        logger.info(f"  {name}: {count}")
    logger.info('-'*30)
    color_database = np.unique(X_test, axis=0)
    # 只保留y_train中出现过的颜色名称，unique_terms为这些唯一颜色名的列表
    name_database = list(set([unique_terms[idx] for idx in y_test]))
    logger.info(f"测试集样本数: {len(X_test)}， 包含 {len(color_database)} 个唯一RGB值， 包含 {len(name_database)} 个唯一颜色名称")

    # 2. 训练模型
    # 记录训练开始时间
    start_time = time.time()
    logger.info("使用预训练Transformer模型训练...")
    # 指定本地模型路径（如果有的话）
    local_model_path = "models-pretrained/bert-base-uncased"  # 修改为你的本地模型路径
    ncf_model, rgb_encoder, name_encoder, rgb_generator = train_pretrained_transformer_model(
        rgb_values=X_train,
        encoded_labels=y_train,
        unique_terms=unique_terms,
        model_name='bert-base-uncased',  # 可选: 'roberta-base', 'distilbert-base-uncased'
        num_epochs=used_num_epochs,
        batch_size= 1024,  
        emb_dim=64,  
        learning_rate= used_learning_rate,  
        save_dir='models-pretrained',
        local_model_path=local_model_path,  # 使用本地模型路径
        use_ncf=use_ncf_architecture
    )
     
    # 记录训练结束时间
    end_time = time.time()
    # 计算训练时间
    training_time = end_time - start_time
    logger.info(f"[Training Time] {training_time/3600:.2f} hours")
    # 3. 测试推理
    logger.info("\n=== 测试推理 ===")
    
    # 如果训练了RGB生成器，则传入进行评估
    test_results = evaluate_model(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, rgb_generator)
    
    if 'topk_accuracy' in test_results:
        logger.info("color2name Top-K 准确率对比:")
        for key in test_results['topk_accuracy']:
            if 'accuracy' in key:
                test_acc = test_results['topk_accuracy'][key]
                logger.info(f"  {key}: {test_acc:.4f}")
    
    if 'name2color_accuracy' in test_results:
        test_cielab_dist = test_results['name2color_accuracy']['cielab_distance_mean']
        logger.info(f"name2color 最小距离推荐:  {test_cielab_dist:.4f}")
        test_cielab_dist = test_results['name2color_accuracy']['cielab_distance_mean_mean']
        logger.info(f"name2color 平均距离推荐:  {test_cielab_dist:.4f}")
    
    logger.info(f"{'='*30}")
    if 'text2color_accuracy' in test_results:
        test_cielab_dist = test_results['text2color_accuracy']['cielab_distance_mean']
        logger.info(f"name2color 生成:  {test_cielab_dist:.4f}")


def compute_topk_accuracy(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, topk_list=[1,3,5], output_sign=False):
    """
    计算Top-K准确率
    """
    rgb_encoder.eval()
    name_encoder.eval()
    if ncf_model is not None:
        ncf_model.eval()
    device = next(rgb_encoder.parameters()).device
    logger.info(f"testing {len(X_test)} samples")
    logger.info(f"testing {len(unique_terms)} unique terms")

    # with torch.no_grad():
    #     name = 'aaaaaa eyes'
    #     origin_name_embding = name_encoder.encode_batch([name])
    #     logger.info(f"{name},\n Name Embding: {origin_name_embding}")
    #     name = 'red'
    #     origin_name_embding = name_encoder.encode_batch([name])
    #     logger.info(f"{name},\n Name Embding: {origin_name_embding}")

    # 分批次预计算所有颜色名称的嵌入，避免显存溢出
    if hasattr(name_encoder, 'encode_batch'):
        if isinstance(unique_terms, np.ndarray):
            unique_terms_list = unique_terms.tolist()
        else:
            unique_terms_list = list(unique_terms)
        
        logger.info(f"🔄 评估阶段：分批次计算 {len(unique_terms_list)} 个颜色名称的embedding...")
        
        # 根据显存大小动态调整批次大小
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            if gpu_memory >= 24:  # 24GB以上显存
                name_batch_size = 1000
            elif gpu_memory >= 16:  # 16GB显存
                name_batch_size = 5000
            else:  # 8GB或更少显存
                name_batch_size = 2500
        else:
            name_batch_size = 5000
            
        logger.info(f"💾 评估批次大小: {name_batch_size}")
        
        # 流式评估：不存储完整相似度矩阵，直接累积topk准确率
        logger.info(f"🔄 开始流式评估，不存储完整相似度矩阵...")
        
        # 转换测试集
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long, device=device)
        
        # 初始化topk统计
        n_samples = len(X_test_tensor)
        y_true = y_test_tensor.cpu().numpy()
        
        # 只维护最大的topk值，其他从它推导 - 使用堆优化
        max_k = max(topk_list)
        logger.info(f"🔄 只维护top{max_k}，其他topk值从中推导")
        
        # 为每个样本维护一个堆，存储(相似度分数, 颜色索引)
        # 使用最小堆，堆顶是最小值，便于快速替换
        import heapq
        topk_heaps = [[] for _ in range(n_samples)]  # 每个样本一个堆
        
        # 记录每个样本当前已填充的topk数量
        topk_counts = np.zeros(n_samples, dtype=np.int32)
        # name_batch_size = 100
        with torch.no_grad():
            # 先计算RGB嵌入（只需要计算一次）
            rgb_embs = rgb_encoder(X_test_tensor)
            
            # 分批次计算名称嵌入和相似度，直接累积topk结果
            for i in range(0, len(unique_terms_list), name_batch_size):
                batch_names = unique_terms_list[i:i+name_batch_size]
                batch_embs = name_encoder.encode_batch(batch_names)
                
                batch_embs = batch_embs.to(device)
               
                # 计算当前批次与所有RGB样本的相似度
                if ncf_model is not None:
                    # NCF模型推理
                    batch_scores = []
                    for j in range(len(X_test_tensor)):
                        rgb_emb = rgb_embs[j].unsqueeze(0).repeat(len(batch_embs), 1)
                        out = ncf_model(rgb_emb, batch_embs)
                        batch_scores.append(out.squeeze().cpu().numpy())
                        
                    batch_similarity = np.stack(batch_scores, axis=0)
                else:
                    # 普通模型推理：计算余弦相似度
                    batch_similarity = torch.matmul(rgb_embs, batch_embs.T).cpu().numpy()
                
                # 向量化处理当前批次的相似度，更新topk预测
                # batch_similarity形状: [n_samples, batch_size]
                # 为每个样本找到当前批次中的topk预测
                # 使用numpy的argpartition进行向量化topk操作
                topk_indices_in_batch = np.argpartition(batch_similarity, -max_k, axis=1)[:, -max_k:]
                
                # 获取对应的相似度分数
                topk_scores_in_batch = np.take_along_axis(batch_similarity, topk_indices_in_batch, axis=1)
                
                # 批量更新topk预测 - 使用堆优化
                for sample_idx in range(n_samples):
                    # 获取当前样本在当前批次中的topk预测
                    for batch_pos in range(max_k):
                        term_idx_in_batch = topk_indices_in_batch[sample_idx, batch_pos]
                        similarity_score = topk_scores_in_batch[sample_idx, batch_pos]
                        global_term_idx = i + term_idx_in_batch
                        
                        # 使用堆操作更新topk预测
                        current_count = topk_counts[sample_idx]
                        if current_count < max_k:
                            # 如果还没满max_k个，直接添加
                            heapq.heappush(topk_heaps[sample_idx], (similarity_score, global_term_idx))
                            topk_counts[sample_idx] += 1
                        else:
                            # 如果已满max_k个，只有更高分数才替换
                            if similarity_score > topk_heaps[sample_idx][0][0]:  # 堆顶是最小值
                                heapq.heapreplace(topk_heaps[sample_idx], (similarity_score, global_term_idx))
                
                # 清理当前批次的embedding
                del batch_embs
                torch.cuda.empty_cache()
                
                # 显示进度和内存使用情况
                if (i // name_batch_size + 1) % 20 == 0:
                    if torch.cuda.is_available():
                        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                        logger.info(f"   流式评估进度: {i + name_batch_size}/{len(unique_terms_list)}, GPU内存: {gpu_memory:.2f}GB")
                    else:
                        logger.info(f"   流式评估进度: {i + name_batch_size}/{len(unique_terms_list)}")
        
        logger.info(f"✅ 流式评估完成，开始计算准确率...")
        logger.info(f"🔄 从top{max_k}推导其他topk值...")
        
        # 计算topk准确率 - 从最大的topk值推导其他topk值
        results = {}
        
        # 从堆中提取所有topk值
        for sample_idx in range(n_samples):
            # 将堆转换为排序后的列表（从高到低）
            heap_data = topk_heaps[sample_idx]
            sorted_data = sorted(heap_data, key=lambda x: x[0], reverse=True)  # 按分数降序排序
            
            # 为每个topk值提取预测结果
            for k in topk_list:
                if k <= len(sorted_data):
                    # 提取topk的索引
                    topk_indices = [item[1] for item in sorted_data[:k]]
                    
                    # 检查真实标签是否在topk预测中
                    if y_true[sample_idx] in topk_indices:
                        if f"top{k}_correct" not in results:
                            results[f"top{k}_correct"] = 0
                        results[f"top{k}_correct"] += 1
                    
                    if k==10 and output_sign:
                        # 将当前sample的top10推荐的color name输出到csv，包含三列：true color，true name，recommend names
                        if not hasattr(compute_topk_accuracy, "csv_written"):
                            # 只写一次表头
                            compute_topk_accuracy.csv_written = False
                        csv_file = "c2n_top10_recommendations.csv"
                        # 只在第一次写入时写表头
                        write_header = not compute_topk_accuracy.csv_written
                        with open(csv_file, "a", newline='', encoding="utf-8") as f:
                            writer = csv.writer(f)
                            if write_header:
                                writer.writerow(["true_color", "true_name", "recommend_names"])
                                compute_topk_accuracy.csv_written = True
                            # 获取真实颜色（前3维），真实名称
                            rgb = X_test[sample_idx]
                            if isinstance(rgb, torch.Tensor):
                                rgb = rgb.cpu().numpy()
                            true_color = ",".join([str(round(float(x), 4)) for x in rgb[:3]])
                            true_label = y_true[sample_idx]
                            if isinstance(unique_terms, np.ndarray):
                                true_name = unique_terms[true_label]
                            else:
                                true_name = list(unique_terms)[true_label]
                            # top10推荐的名称
                            top10_indices = [item[1] for item in sorted_data[:10]]
                            recommend_names = []
                            for idx in top10_indices:
                                if isinstance(unique_terms, np.ndarray):
                                    recommend_names.append(unique_terms[idx])
                                else:
                                    recommend_names.append(list(unique_terms)[idx])
                            recommend_names_str = "|".join(recommend_names)
                            writer.writerow([true_color, true_name, recommend_names_str])
        
        # 计算准确率
        for k in topk_list:
            correct = results.get(f"top{k}_correct", 0)
            acc = correct / n_samples
            results[f"top{k}_accuracy"] = acc
            logger.info(f"Top-{k} 准确率: {acc:.4f}")
        
        # 强制垃圾回收
        import gc
        gc.collect()
        logger.info("🧹 已清理内存和GPU缓存")

    return results

def compute_name2color_accuracy_simple(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, top_k=5):
    """
    计算名称到颜色的推荐准确率
    直接全部一块计算，不使用分批处理
    
    Args:
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        ncf_model: NCF模型（可选）
        X_test: 测试集RGB值
        y_test: 测试集标签
        unique_terms: 唯一颜色名称列表
        unique_colors: 唯一颜色值列表
        top_k: 推荐数量
    
    Returns:
        results: 包含准确率统计的字典
    """
    logger.info(f"=== 计算名称到颜色推荐准确率（简化版本）===")
    
    rgb_encoder.eval()
    name_encoder.eval()
    if ncf_model is not None:
        ncf_model.eval()
    
    device = next(rgb_encoder.parameters()).device
    logger.info(f"使用设备: {device}")
    
    # 确保unique_terms是列表格式
    if isinstance(unique_terms, np.ndarray):
        unique_terms_list = unique_terms.tolist()
    else:
        unique_terms_list = list(unique_terms)
    
    total_samples = len(X_test)
    logger.info(f"开始评估 {total_samples} 个样本...")
    logger.info(f"unique_colors包含 {len(unique_colors)} 个不同的RGB值")
    logger.info(f"unique_terms包含 {len(unique_terms_list)} 个不同的颜色名称")
    y_test_names = list(set([unique_terms_list[idx] for idx in y_test]))
    logger.info(f"y_test包含 {len(y_test_names)} 个不同的颜色名称")

    # 直接计算所有RGB的embedding（不做分批处理）
    logger.info("🔄 计算所有RGB的embedding...")
    start_time = time.time()
    
    # 将unique_colors转换为numpy数组
    if isinstance(unique_colors[0], tuple):
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    else:
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    
    # 处理RGB特征
    rgb_features = []
    for rgb in unique_colors_array:
        if use_16d_feature:
            # 使用16维特征
            if len(rgb) == 3:
                feature = rgb_to_16d_feature(rgb)
            elif len(rgb) == 16:
                feature = rgb
            else:
                logger.info(f"警告: 意外的RGB维度 {len(rgb)}，使用简单填充")
                feature = list(rgb[:3]) + [0.0] * 13
        else:
            # 使用3维RGB特征
            if len(rgb) >= 3:
                feature = rgb[:3]  # 只取前3维
            else:
                logger.info(f"警告: RGB维度不足 {len(rgb)}，使用零填充")
                feature = list(rgb) + [0.0] * (3 - len(rgb))
        
        rgb_features.append(feature)
    
    rgb_features = np.array(rgb_features, dtype=np.float32)
    rgb_tensor = torch.tensor(rgb_features, dtype=torch.float32, device=device)
    
    # 直接计算所有RGB embedding
    with torch.no_grad():
        all_rgb_embeddings = rgb_encoder(rgb_tensor)
    
    logger.info(f"✅ RGB embedding计算完成，形状: {all_rgb_embeddings.shape}")
    logger.info(f"⏱️ RGB embedding计算耗时: {time.time() - start_time:.2f}秒")
    
    # 直接计算所有颜色名称的embedding
    logger.info("🔄 计算所有颜色名称的embedding...")
    start_time = time.time()
    
    with torch.no_grad():
        all_name_embeddings = name_encoder.encode_batch(unique_terms_list)
    
    logger.info(f"✅ 名称embedding计算完成，形状: {all_name_embeddings.shape}")
    logger.info(f"⏱️ 名称embedding计算耗时: {time.time() - start_time:.2f}秒")
    
    # 直接计算所有测试样本的准确率
    logger.info("🔄 开始计算准确率...")
    start_time = time.time()
    
    correct_predictions = 0
    total_predictions = 0
    cielab_distances = []
    for i in range(len(X_test)):
        # 获取当前样本的真实颜色名称
        true_name_idx = y_test[i]
        true_name = unique_terms_list[true_name_idx]
        
        # 获取当前样本的真实RGB值
        true_rgb = X_test[i][:3]  # 取前3维作为RGB
        
        # 使用NCF模型计算相似度分数
        if ncf_model is not None:
            # 获取名称embedding
            name_emb = all_name_embeddings[true_name_idx:true_name_idx+1]  # (1, emb_dim)
            
            # 计算与所有RGB的相似度
            all_scores = []
            with torch.no_grad():
                # 方法1：批量计算（推荐）
                # 将name_emb扩展到与all_rgb_embeddings相同的batch size
                name_emb_expanded = name_emb.expand(all_rgb_embeddings.size(0), -1)  # (num_rgb, emb_dim)
                scores = ncf_model(all_rgb_embeddings, name_emb_expanded)  # (num_rgb, 1)
                all_scores = scores.squeeze().cpu().numpy()  # (num_rgb,)
            
            # 获取top-k推荐
            top_indices = np.argsort(all_scores)[::-1][:top_k]
            
        else:
            # 使用余弦相似度
            name_emb = all_name_embeddings[true_name_idx:true_name_idx+1]  # (1, emb_dim)
            name_emb_expanded = name_emb.expand(all_rgb_embeddings.size(0), -1)  # (num_rgb, emb_dim)
            similarities = torch.sum(name_emb_expanded * all_rgb_embeddings, dim=1)
            top_indices = torch.topk(similarities, top_k).indices.cpu().numpy()
        
        # 检查真实RGB是否在top-k推荐中
        true_rgb_found = False
        for idx in top_indices:
            recommended_rgb = unique_colors_array[idx][:3]
            # 使用欧几里得距离判断RGB是否匹配
            rgb_distance = np.sqrt(np.sum((true_rgb - recommended_rgb) ** 2))
            if rgb_distance < 0.01:  # 阈值可调整
                true_rgb_found = True
                break
        # 获取第一个推荐（最高分数）
        pred_rgb_idx = top_indices[0]
        pred_rgb = unique_colors_array[pred_rgb_idx]
        
        
        # 获取真实LAB值
        true_lab = rgb_to_lab(true_rgb)
        
        # 将预测的RGB转换为LAB
        pred_lab = rgb_to_lab(pred_rgb[:3])
        
        # 计算CIELAB距离
        cielab_dist = np.sqrt(np.sum((np.array(pred_lab) - np.array(true_lab))**2))
        cielab_distances.append(cielab_dist)
        
        if true_rgb_found:
            correct_predictions += 1
        total_predictions += 1
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            logger.info(f"  进度: {i + 1}/{total_samples}, 当前准确率: {correct_predictions/total_predictions:.4f}")
    
    # 过滤掉无效的CIELAB距离
    valid_cielab_dists = [d for d in cielab_distances if d != float('inf') and not np.isnan(d)]
    
    # 计算最终准确率
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    logger.info(f"✅ 准确率计算完成")
    logger.info(f"⏱️ 准确率计算耗时: {time.time() - start_time:.2f}秒")
    logger.info(f"📊 最终结果:")
    logger.info(f"  总样本数: {total_samples}")
    logger.info(f"  正确预测数: {correct_predictions}")
    logger.info(f"  Top-{top_k} 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    logger.info(f"  CIELAB距离 - 平均: {np.mean(valid_cielab_dists):.4f}, 标准差: {np.std(valid_cielab_dists):.4f}")
    
    # 返回结果
    results = {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        f'top_{top_k}_accuracy': accuracy,
        'accuracy_percentage': accuracy * 100,
        'cielab_distance_mean': np.mean(valid_cielab_dists),
        'cielab_distance_std': np.std(valid_cielab_dists),
        'cielab_distance_sum': np.sum(valid_cielab_dists)
    }
    
    return results

def compute_name2color_accuracy(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, top_k=5, topk_list=[1,3, 5, 10, 15], output_sign=False):
    """
    计算名称到颜色的推荐准确率（Name-to-Color Accuracy）
    参考text2color_lstm_pytorch.py中的eval_cielab_distance评估方法
    
    Args:
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        ncf_model: NCF模型（可选）
        X_test: 测试集RGB值
        y_test: 测试集标签
        unique_terms: 唯一颜色名称列表
        unique_colors: 唯一颜色值列表
        top_k: 主要推荐数量（用于计算主要指标）
        topk_list: 要计算的top-k准确率列表，如[1,3,5,10]
    
    Returns:
        results: 包含准确率统计的字典
    """
    logger.info(f"=== 计算名称到颜色推荐准确率 ===")
    
    rgb_encoder.eval()
    name_encoder.eval()
    if ncf_model is not None:
        ncf_model.eval()
    
    device = next(rgb_encoder.parameters()).device
    logger.info(f"使用设备: {device}")
    
    # 确保unique_terms是列表格式
    if isinstance(unique_terms, np.ndarray):
        unique_terms_list = unique_terms.tolist()
    else:
        unique_terms_list = list(unique_terms)
    
    # 初始化统计变量
    total_samples = len(X_test)
    cielab_distances = []
    cielab_distances_mean = []
    correct_predictions = 0
    total_predictions = 0
    
    # 初始化多topk统计
    max_topk = 1000
    topk_correct_counts = {k: 0 for k in topk_list}
    topk_cielab_distances = {k: [] for k in topk_list}
    topk_cielab_distances_mean = {k: [] for k in topk_list}
    
    logger.info(f"开始评估 {total_samples} 个样本...")
    logger.info(f"unique_colors包含 {len(unique_colors)} 个不同的RGB值")
    
    # 🚀 优化1: 预计算所有RGB的embedding，避免重复计算
    logger.info("🔄 预计算所有RGB的embedding...")
    start_time = time.time()
    
    # 将unique_colors转换为numpy数组并移到设备上
    if isinstance(unique_colors[0], tuple):
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    else:
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    
    # 分批次计算RGB embedding，避免显存溢出
    rgb_batch_size = 100000  # 根据显存调整
    all_rgb_embeddings = []
    
    for i in range(0, len(unique_colors_array), rgb_batch_size):
        start_idx = i
        end_idx = min(start_idx + rgb_batch_size, len(unique_colors_array))
        batch_colors = unique_colors_array[start_idx:end_idx]
        # 根据use_16d_feature设置处理RGB特征
        batch_features = []
        for rgb in batch_colors:
            if use_16d_feature:
                # 使用16维特征
                if len(rgb) == 3:
                    feature = rgb_to_16d_feature(rgb)
                elif len(rgb) == 16:
                    feature = rgb
                else:
                    logger.info(f"警告: 意外的RGB维度 {len(rgb)}，使用简单填充")
                    feature = list(rgb[:3]) + [0.0] * 13
            else:
                # 使用3维RGB特征
                if len(rgb) >= 3:
                    feature = rgb[:3]  # 只取前3维
                else:
                    logger.info(f"警告: RGB维度不足 {len(rgb)}，使用零填充")
                    feature = list(rgb) + [0.0] * (3 - len(rgb))
            
            batch_features.append(feature)
        batch_features = np.array(batch_features, dtype=np.float32)
        # logger.info(f"first 5 samples of batch_features: {batch_features[:5]}")
        # 转换为特征张量
        batch_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
    
        with torch.no_grad():
            batch_embeddings = rgb_encoder(batch_tensor)
            all_rgb_embeddings.append(batch_embeddings.cpu())
        
        if (i // rgb_batch_size + 1) % 10 == 0:
            logger.info(f"  RGB embedding进度: {i + rgb_batch_size}/{len(unique_colors_array)}")
    
    all_rgb_embeddings = torch.cat(all_rgb_embeddings, dim=0).to(device)
    logger.info(f"✅ RGB embedding预计算完成，形状: {all_rgb_embeddings.shape}")
    logger.info(f"⏱️ RGB embedding预计算耗时: {time.time() - start_time:.2f}秒")
    
    # 🚀 优化2: 只预计算y_test中出现的颜色名称的embedding
    logger.info("🔄 只预计算y_test中出现的颜色名称的embedding...")
    start_time = time.time()
    
    # 获取y_test中出现的唯一颜色名称索引
    unique_test_indices = np.unique(y_test)
    unique_test_names = [unique_terms_list[idx] for idx in unique_test_indices]
    logger.info(f"y_test中包含 {len(unique_test_indices)} 个唯一的颜色名称索引")
    logger.info(f"实际需要编码的颜色名称数量: {len(unique_test_names)}")
    logger.info(unique_test_indices)
    
    name_batch_size = min(10000, len(unique_test_names))
    all_name_embeddings = []
    
    for i in range(0, len(unique_test_names), name_batch_size):
        start_idx = i
        end_idx = min(start_idx + name_batch_size, len(unique_test_names))
        batch_names = unique_test_names[start_idx:end_idx]
        with torch.no_grad():
            batch_embeddings = name_encoder.encode_batch(batch_names)
            all_name_embeddings.append(batch_embeddings.cpu())
        
        if (i // name_batch_size + 1) % 20 == 0:
            logger.info(f"  名称embedding进度: {i + name_batch_size}/{len(unique_test_names)}")
    
    all_name_embeddings = torch.cat(all_name_embeddings, dim=0).to(device)
    logger.info(f"✅ 名称embedding预计算完成，形状: {all_name_embeddings.shape}")
    logger.info(f"⏱️ 名称embedding预计算耗时: {time.time() - start_time:.2f}秒")
    
    # 🚀 优化3: 使用分块计算避免大矩阵内存爆炸
    logger.info("🔄 开始分块评估，避免内存不足...")
    start_time = time.time()
    
    rgb_chunk_size = 100000  # 每次处理10万个RGB
    
    logger.info(f"测试样本数量: {total_samples}, RGB分块大小: {rgb_chunk_size}")
    n_rgb_chunks = (len(unique_colors_array) + rgb_chunk_size - 1) // rgb_chunk_size
    logger.info(f"RGB分块数量: {n_rgb_chunks}")
    
    # 创建索引映射：从原始索引到新索引
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_test_indices)}

    # 对每个测试样本进行处理
    for i in range(len(X_test)):
        # 获取当前样本的真实颜色名称
        true_name_idx = y_test[i]
        true_name = unique_terms_list[true_name_idx]
        # 获取名称embedding
        name_emb = all_name_embeddings[index_mapping[true_name_idx]:index_mapping[true_name_idx]+1]  # (1, emb_dim)
        # 获取当前样本的真实RGB值
        true_rgb = X_test[i][:3]  # 取前3维作为RGB

        # 为当前样本维护top-k结果（使用最大的topk值）
        sample_top_k_values = torch.full((max_topk,), float('-inf'), device=device)
        sample_top_k_indices = torch.zeros((max_topk,), dtype=torch.long, device=device)
        
        # 分块处理RGB，避免大矩阵
        with torch.no_grad():
            for rgb_chunk_idx in range(n_rgb_chunks):
                rgb_start = rgb_chunk_idx * rgb_chunk_size
                rgb_end = min(rgb_start + rgb_chunk_size, len(unique_colors_array))
                
                # 获取当前RGB块的embedding
                rgb_chunk_embeddings = all_rgb_embeddings[rgb_start:rgb_end]
                
                # 计算当前样本与当前RGB块的相似度
                if ncf_model is not None:
                    # 使用NCF模型计算相似度
                    name_emb_expanded = name_emb.repeat(len(rgb_chunk_embeddings), 1)
                    chunk_scores = ncf_model(rgb_chunk_embeddings, name_emb_expanded)
                    chunk_similarity = chunk_scores.squeeze()
                else:
                    # 使用余弦相似度
                    chunk_similarity = F.cosine_similarity(name_emb, rgb_chunk_embeddings, dim=1)
                
                # 更新当前样本的top-k结果
                # 获取当前chunk中的top-k
                chunk_values, chunk_indices = torch.topk(chunk_similarity, k=min(max_topk, chunk_similarity.size(0)))
                
                # 与全局top-k比较，保留最大的
                combined_values = torch.cat([sample_top_k_values, chunk_values])
                combined_indices = torch.cat([sample_top_k_indices, chunk_indices + rgb_start])
                
                # 找到全局top-k
                global_top_k_values, global_top_k_indices = torch.topk(combined_values, k=max_topk)
                sample_top_k_values = global_top_k_values
                sample_top_k_indices = combined_indices[global_top_k_indices]
                
                # 清理当前chunk的显存
                del chunk_similarity, rgb_chunk_embeddings
                if ncf_model is not None:
                    del name_emb_expanded, chunk_scores
                torch.cuda.empty_cache()
        
        # 获取推荐的RGB值并计算准确率
        recommended_indices = sample_top_k_indices.cpu().numpy()
        
        # 获取真实LAB值
        true_lab = rgb_to_lab(true_rgb)
        
        # 计算所有推荐RGB的LAB值
        pred_rgbs = unique_colors_array[recommended_indices][:, :3]
        # 对pred_rgbs中的颜色进行去重：从predlab0开始，与去重数组中的所有颜色进行比较，如果lab距离小于1，则不选择，否则加入去重数组
        # 批量计算LAB值，使用向量化去重
        # 批量转换所有RGB到LAB
        pred_labs_batch = np.array([rgb_to_lab(rgb) for rgb in pred_rgbs])
        
        # 使用向量化操作进行去重
        unique_pred_rgbs = []
        unique_pred_labs = []
        
        for ii, (rgb, lab) in enumerate(zip(pred_rgbs, pred_labs_batch)):
            if len(unique_pred_labs) == 0:
                # 第一个颜色直接添加
                unique_pred_rgbs.append(rgb)
                unique_pred_labs.append(lab)
            else:
                # 向量化计算与所有已去重颜色的距离
                existing_labs = np.array(unique_pred_labs)
                distances = np.sqrt(np.sum((lab - existing_labs) ** 2, axis=1))
                
                # 如果最小距离大于阈值，则添加
                if np.min(distances) >= 2.3:
                    unique_pred_rgbs.append(rgb)
                    unique_pred_labs.append(lab)
                if len(unique_pred_rgbs) >= 10:
                    # logger.info(f"推荐颜色数量超过10个: {ii}")
                    break
        # logger.info(f"推荐颜色数量: {len(unique_pred_rgbs)}")
        if len(unique_pred_rgbs) < 10:
            logger.info(f"第{i}个样本, 警告: 推荐颜色数量不足10个: {len(unique_pred_rgbs)}")
        pred_labs = np.array(unique_pred_labs)
        lab_dists = np.sqrt(np.sum((pred_labs - true_lab) ** 2, axis=1))

        # 将unique_pred_rgbs输出到csv，三列分别是true color，true name，recommend colors
        if output_sign:
            csv_file = "n2c_top10_recommendations.csv"
            # 只在第一次写入时写表头
            if not hasattr(compute_name2color_accuracy_simple, "csv_written"):
                compute_name2color_accuracy_simple.csv_written = False
            write_header = not compute_name2color_accuracy_simple.csv_written
            with open(csv_file, "a", newline='', encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["true_color", "true_name", "recommend_colors"])
                    compute_name2color_accuracy_simple.csv_written = True
                # true color
                rgb = true_rgb
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.cpu().numpy()
                true_color = ",".join([str(round(float(x), 4)) for x in rgb[:3]])
                # true name
                if isinstance(unique_terms, np.ndarray):
                    true_name = unique_terms[y_test[i]]
                else:
                    true_name = list(unique_terms)[y_test[i]]
                # recommend colors
                recommend_colors = []
                for rgb_pred in unique_pred_rgbs:
                    recommend_colors.append(",".join([str(round(float(x), 4)) for x in rgb_pred[:3]]))
                recommend_colors_str = "|".join(recommend_colors)
                writer.writerow([true_color, true_name, recommend_colors_str])
        
        # 为每个topk计算准确率和CIELAB距离
        for k in topk_list:
            if k <= len(lab_dists):
                # 检查真实RGB是否在top-k推荐中
                true_rgb_found = False
                for idx in range(min(k, len(lab_dists))):
                    if lab_dists[idx] < 1:  # 阈值可调整
                        true_rgb_found = True
                        break
        
                # 统计准确率
                if true_rgb_found:
                    topk_correct_counts[k] += 1
                
                # 计算CIELAB距离（使用top-k中的最小距离）
                min_lab_dist = np.min(lab_dists[:k])
                topk_cielab_distances[k].append(min_lab_dist)
                mean_lab_dist = np.mean(lab_dists[:k])
                topk_cielab_distances_mean[k].append(mean_lab_dist)

                if k == top_k:
                    # 统计主要准确率
                    if true_rgb_found:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    # 计算主要CIELAB距离
                    cielab_distances.append(min_lab_dist)
                    
                    # 计算平均RGB的CIELAB距离
                    cielab_distances_mean.append(mean_lab_dist)
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            logger.info(f"  进度: {i + 1}/{total_samples}, 当前准确率: {correct_predictions/total_predictions:.4f}")
    
    logger.info(f"✅ 分块评估完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 过滤掉无效的CIELAB距离
    valid_cielab_dists = [d for d in cielab_distances if d != float('inf')]
    valid_cielab_dists_mean = [d for d in cielab_distances_mean if d != float('inf')]
    
    # 计算主要准确率
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # 计算多topk准确率
    topk_accuracies = {}
    for k in topk_list:
        topk_accuracies[f'top_{k}_accuracy'] = topk_correct_counts[k] / total_samples if total_samples > 0 else 0
    
    # 计算多topk CIELAB距离统计
    topk_cielab_stats = {}
    for k in topk_list:
        valid_dists = [d for d in topk_cielab_distances[k] if d != float('inf') and not np.isnan(d)]
        valid_dists_mean = [d for d in topk_cielab_distances_mean[k] if d != float('inf') and not np.isnan(d)]
        if valid_dists:
            topk_cielab_stats[f'top_{k}_cielab_mean'] = np.mean(valid_dists)
            topk_cielab_stats[f'top_{k}_cielab_std'] = np.std(valid_dists)
            topk_cielab_stats[f'top_{k}_cielab_mean_mean'] = np.mean(valid_dists_mean)
            topk_cielab_stats[f'top_{k}_cielab_std_mean'] = np.std(valid_dists_mean)
        else:
            topk_cielab_stats[f'top_{k}_cielab_mean'] = float('inf')
            topk_cielab_stats[f'top_{k}_cielab_std'] = 0.0
            topk_cielab_stats[f'top_{k}_cielab_mean_mean'] = float('inf')
            topk_cielab_stats[f'top_{k}_cielab_std_mean'] = 0.0

    results = {
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        f'top_{top_k}_accuracy': accuracy,
        'accuracy': accuracy,
        'accuracy_percentage': accuracy * 100,
        'cielab_distance_mean': np.mean(valid_cielab_dists) if valid_cielab_dists else float('inf'),
        'cielab_distance_std': np.std(valid_cielab_dists) if valid_cielab_dists else 0.0,
        'cielab_distance_mean_mean': np.mean(valid_cielab_dists_mean) if valid_cielab_dists_mean else float('inf'),
        'cielab_distance_mean_std': np.std(valid_cielab_dists_mean) if valid_cielab_dists_mean else 0.0,
        **topk_accuracies,
        **topk_cielab_stats
    }
    
    # 打印结果
    logger.info(f"\n=== 名称到颜色推荐准确率结果 ===")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"正确预测数: {correct_predictions}")
    logger.info(f"Top-{top_k} 准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 打印多topk准确率
    logger.info(f"\n=== 多Top-K准确率 ===")
    for k in topk_list:
        acc = topk_accuracies[f'top_{k}_accuracy']
        logger.info(f"Top-{k} 准确率: {acc:.4f} ({acc*100:.2f}%)")
    
    # 打印多topk CIELAB距离
    logger.info(f"\n=== 多Top-K CIELAB距离统计 ===")
    for k in topk_list:
        mean_dist = topk_cielab_stats[f'top_{k}_cielab_mean']
        std_dist = topk_cielab_stats[f'top_{k}_cielab_std']
        mean_dist_mean = topk_cielab_stats[f'top_{k}_cielab_mean_mean']
        std_dist_mean = topk_cielab_stats[f'top_{k}_cielab_std_mean']
        if mean_dist != float('inf') and mean_dist_mean != float('inf'):
            logger.info(f"Top-{k} CIELAB min距离 - 平均: {mean_dist:.4f}, 标准差: {std_dist:.4f}")
            logger.info(f"Top-{k} CIELAB avg距离 - 平均: {mean_dist_mean:.4f}, 标准差: {std_dist_mean:.4f}")
        else:
            logger.info(f"Top-{k} CIELAB距离 - 无有效数据")
    
    logger.info(f"\n=== 主要指标 Top-{top_k} ===")
    logger.info(f"CIELAB最小距离 - 平均: {results['cielab_distance_mean']:.4f}, 标准差: {results['cielab_distance_std']:.4f}")
    logger.info(f"CIELAB平均距离 - 平均: {results['cielab_distance_mean_mean']:.4f}, 标准差: {results['cielab_distance_mean_std']:.4f}")
    
    # 🚀 优化5: 清理GPU内存
    logger.info("🧹 清理GPU内存...")
    del all_rgb_embeddings, all_name_embeddings
    torch.cuda.empty_cache()
    
    return results

def compute_text2color_accuracy(name_encoder, rgb_generator, X_test, y_test, unique_terms, use_cielab=True):
    """
    计算文本到颜色的生成准确率（Text-to-Color Accuracy）
    测试RGB生成器的性能：给定颜色名称，生成对应的RGB值，与真实RGB值比较
    
    Args:
        rgb_encoder: RGB编码器（可选，用于对比）
        name_encoder: 名称编码器
        rgb_generator: RGB生成器
        X_test: 测试集RGB值
        y_test: 测试集标签
        unique_terms: 唯一颜色名称列表
        use_cielab: 是否使用CIELAB颜色空间计算距离
    
    Returns:
        results: 包含准确率统计的字典
    """
    logger.info(f"=== 计算文本到颜色生成准确率 ===")

    name_encoder.eval()
    rgb_generator.eval()
    device = next(rgb_generator.parameters()).device
    logger.info(f"使用设备: {device}")
    
    # 确保unique_terms是列表格式
    if isinstance(unique_terms, np.ndarray):
        unique_terms_list = unique_terms.tolist()
    else:
        unique_terms_list = list(unique_terms)
    
    # 初始化统计变量
    total_samples = len(X_test)
    rgb_distances = []
    cielab_distances = []
    
    logger.info(f"开始评估 {total_samples} 个样本...")
    
    # 🚀 优化1: 预计算所有颜色名称的embedding，避免重复计算
    logger.info("🔄 预计算所有颜色名称的embedding...")
    start_time = time.time()
    
    # 获取y_test中出现的唯一颜色名称索引
    unique_test_indices = np.unique(y_test)
    unique_test_names = [unique_terms_list[idx] for idx in unique_test_indices]
    logger.info(f"y_test中包含 {len(unique_test_indices)} 个唯一的颜色名称索引")
    logger.info(f"实际需要编码的颜色名称数量: {len(unique_test_names)}")
    
    # 创建索引映射：从原始索引到新索引
    index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_test_indices)}
    
    # 分批次计算名称embedding，避免显存溢出
    name_batch_size = 1000  # 根据显存调整
    all_name_embeddings = []
    
    for i in range(0, len(unique_test_names), name_batch_size):
        batch_names = unique_test_names[i:i+name_batch_size]
        with torch.no_grad():
            batch_embeddings = name_encoder(batch_names)
            all_name_embeddings.append(batch_embeddings.cpu())
        
        if (i // name_batch_size + 1) % 20 == 0:
            logger.info(f"  名称embedding进度: {i + name_batch_size}/{len(unique_test_names)}")
    
    all_name_embeddings = torch.cat(all_name_embeddings, dim=0).to(device)
    logger.info(f"✅ 名称embedding预计算完成，形状: {all_name_embeddings.shape}")
    logger.info(f"⏱️ 名称embedding预计算耗时: {time.time() - start_time:.2f}秒")
    
    # 🚀 优化2: 使用分块计算避免大矩阵内存爆炸
    logger.info("🔄 开始分块评估，避免内存不足...")
    start_time = time.time()
    
    # 🚨 关键优化：根据显存动态调整批次大小
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        if gpu_memory >= 24:  # 24GB以上显存
            test_batch_size = 200
        elif gpu_memory >= 16:  # 16GB显存
            test_batch_size = 100
        else:  # 8GB或更少显存
            test_batch_size = 50
    else:
        test_batch_size = 50
    
    logger.info(f"💾 GPU显存: {gpu_memory:.1f}GB, 测试批次大小: {test_batch_size}")
    
    n_test_batches = (total_samples + test_batch_size - 1) // test_batch_size
    
    for batch_idx in range(n_test_batches):
        start_idx = batch_idx * test_batch_size
        end_idx = min(start_idx + test_batch_size, total_samples)
        
        batch_X = X_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]
        
        # 获取批次中的颜色名称索引
        batch_name_indices = batch_y
        
        with torch.no_grad():
            # 获取当前批次的名称embedding，使用索引映射
            batch_mapped_indices = [index_mapping[idx] for idx in batch_name_indices]
            batch_name_embs = all_name_embeddings[batch_mapped_indices]  # [batch_size, emb_dim]
            
            # 使用RGB生成器生成RGB值
            generated_rgb = rgb_generator(batch_name_embs)  # [batch_size, 3]
            
            # 获取真实的RGB值（前3维）
            true_rgb = torch.tensor(batch_X[:, :3], dtype=torch.float32, device=device)
            
            # 计算RGB欧几里得距离
            rgb_dist = torch.sqrt(torch.sum((generated_rgb - true_rgb) ** 2, dim=1))
            rgb_distances.extend(rgb_dist.cpu().numpy())
            
            # 计算CIELAB距离（如果启用）
            if use_cielab:
                for i in range(len(batch_name_indices)):
                    try:
                        # 获取生成的RGB值
                        pred_rgb = generated_rgb[i].cpu().numpy()
                        
                        # 获取真实RGB值
                        true_rgb_sample = true_rgb[i].cpu().numpy()
                        
                        # 转换为CIELAB颜色空间
                        pred_lab = rgb_to_lab(pred_rgb)
                        true_lab = rgb_to_lab(true_rgb_sample)
                        
                        # 计算CIELAB距离
                        cielab_dist = np.sqrt(np.sum((np.array(pred_lab) - np.array(true_lab))**2))
                        cielab_distances.append(cielab_dist)
                        
                    except Exception as e:
                        logger.info(f"⚠️ 样本 {i} CIELAB转换失败: {e}")
                        # 使用RGB距离作为fallback
                        cielab_distances.append(rgb_dist[i].cpu().item())
        
        # 显示进度
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  测试批次进度: {batch_idx + 1}/{n_test_batches}")
    
    logger.info(f"✅ 分块评估完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 过滤掉无效的距离值
    valid_rgb_dists = [d for d in rgb_distances if d != float('inf') and not np.isnan(d)]
    valid_cielab_dists = [d for d in cielab_distances if d != float('inf') and not np.isnan(d)]
    
    # 计算统计结果
    results = {
        'total_samples': total_samples,
        'rgb_distance_mean': np.mean(valid_rgb_dists),
        'rgb_distance_std': np.std(valid_rgb_dists),
        'rgb_distance_sum': np.sum(valid_rgb_dists),
        'rgb_distance_min': np.min(valid_rgb_dists),
        'rgb_distance_max': np.max(valid_rgb_dists)
    }
    
    if use_cielab and valid_cielab_dists:
        results.update({
            'cielab_distance_mean': np.mean(valid_cielab_dists),
            'cielab_distance_std': np.std(valid_cielab_dists),
            'cielab_distance_sum': np.sum(valid_cielab_dists),
            'cielab_distance_min': np.min(valid_cielab_dists),
            'cielab_distance_max': np.max(valid_cielab_dists)
        })
    
    # 打印结果
    logger.info(f"\n=== 文本到颜色生成准确率结果 ===")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"RGB距离 - 平均: {results['rgb_distance_mean']:.4f}, 标准差: {results['rgb_distance_std']:.4f}")
    logger.info(f"RGB距离 - 最小: {results['rgb_distance_min']:.4f}, 最大: {results['rgb_distance_max']:.4f}")
    logger.info(f"RGB距离 - 总和: {results['rgb_distance_sum']:.4f}")
    
    if use_cielab and valid_cielab_dists:
        logger.info(f"CIELAB距离 - 平均: {results['cielab_distance_mean']:.4f}, 标准差: {results['cielab_distance_std']:.4f}")
        logger.info(f"CIELAB距离 - 最小: {results['cielab_distance_min']:.4f}, 最大: {results['cielab_distance_max']:.4f}")
        logger.info(f"CIELAB距离 - 总和: {results['cielab_distance_sum']:.4f}")
    
    # 🚀 优化3: 清理GPU内存
    logger.info("🧹 清理GPU内存...")
    del all_name_embeddings
    torch.cuda.empty_cache()
    
    return results

def evaluate_model(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, rgb_generator=None):
    """
    评估模型
    
    Args:
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        ncf_model: NCF模型（可选）
        X_test: 测试集RGB值
        y_test: 测试集标签
        unique_terms: 唯一颜色名称列表
        unique_colors: 唯一颜色值列表
        rgb_generator: RGB生成器（可选，用于测试生成性能）
    """
    logger.info("=== 评估模型 ===")
    # 在计算embedding之前设置模型为评估模式
    rgb_encoder.eval()
    name_encoder.eval()
    if ncf_model is not None:
        ncf_model.eval()
    if rgb_generator is not None:
        rgb_generator.eval()
    device = next(rgb_encoder.parameters()).device
    logger.info(f"使用设备: {device}")

    test_color = [[[1,0,0], 'red'], [[1,0,0], 'green'], [[0.5,0,0], 'red'], [[0,1,0],'red']]
    for color in test_color:
        # 根据use_16d_feature设置决定输入格式
        if use_16d_feature:
            # 将RGB转换为16维特征输入
            rgb_input = rgb_to_16d_feature(np.array(color[0], dtype=np.float32))
            logger.info(f"RGB 16D: {rgb_input}")
        else:
            # 直接使用3维RGB输入
            rgb_input = np.array(color[0], dtype=np.float32)
            logger.info(f"RGB 3D: {rgb_input}")
        
        rgb_tensor = torch.tensor([rgb_input], dtype=torch.float32, device=device)
        with torch.no_grad():
            rgb_embding = rgb_encoder(rgb_tensor)
            name_embding = name_encoder([color[1]])
            if ncf_model is not None:
                score = ncf_model(rgb_embding, name_embding)
            else:
                score = torch.sum(rgb_embding * name_embding, dim=1)
        # logger.info("name_embding:", name_embding)
        # logger.info("rgb_embding:", rgb_embding)
        logger.info(f"RGB: {color[0]}, Name: {color[1]}, Score: {score.item():.4f}")
        # logger.info('-----------------')
        # rgbs, scores = recommend_rgb_colors(ncf_model, rgb_encoder, name_encoder, color[1], unique_colors, top_k=5)
        # logger.info(f"Name:{color[1]}, 推荐颜色: {rgbs}, 对应RGB值: {rgbs*255}, 置信度: {scores}")
        # names, scores = recommend_color_names(ncf_model, rgb_encoder, name_encoder, color[0], unique_terms, top_k=5)
        # logger.info(f"RGB:{color[0]}, 推荐颜色名称: {names}, 置信度: {scores}")
        # rgb = rgb_generator(name_embding)
        # logger.info(f"Name: {color[1]}, 推荐颜色: {rgb}, 对应RGB值: {rgb*255}")
        logger.info('-----------------')

    # 计算Top-N 准确率
    # 基于rgb_encoder, name_encoder, ncf_model，计算Top-N（Top-1, Top-3, Top-5）准确率
    topk_list = [1,3,5,10]
    topk_results = compute_topk_accuracy(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, topk_list)
    
    # 计算名称到颜色的推荐准确率
    logger.info(f"\n{'='*60}")
    name2color_results = compute_name2color_accuracy(
        rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, 
        top_k=5
    )
    # logger.info(f"\n{'='*60}")
    # name2color_results = compute_name2color_accuracy_simple(
    #     rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, 
    #     top_k=5
    # )
    
    # 计算文本到颜色的生成准确率（如果提供了RGB生成器）
    logger.info(f"\n{'='*60}")
    text2color_results = compute_text2color_accuracy(name_encoder, rgb_generator, X_test, y_test, unique_terms)
    
    # 合并结果
    all_results = {
        'topk_accuracy': topk_results,
        'name2color_accuracy': name2color_results,
        'text2color_accuracy': text2color_results
    }
    
    return all_results
    test_colors = [
        ([0.5, 0.2, 0.8], "purple"),
        ([0.8, 0.2, 0.2], "red"),
        # ([0.2, 0.8, 0.2], "green"),
        # ([0.2, 0.2, 0.8], "blue"),
        # ([1.0, 1.0, 0.0], "yellow"),
        # ([0.0, 0.0, 0.0], "black"),
        # ([1.0, 1.0, 1.0], "white"),
        # ([0.5, 0.5, 0.5], "grey"),
        # ([1.0, 0.5, 0.0], "orange"),
        ([0.5, 0.0, 0.5], "magenta")
    ]

    for rgb, color_name in test_colors:
        names, scores = recommend_color_names_ncf(
                ncf_model, rgb_encoder, name_encoder, rgb, unique_terms, top_k=5
            )
        logger.info(f"测试颜色: {rgb}，真实颜色名称: {color_name}, 推荐颜色名称: {names}, 置信度: {scores}")
    
    for rgb, color_name in test_colors:
        rgb_values, scores = recommend_rgb_colors_ncf(
                ncf_model, rgb_encoder, name_encoder, color_name, X_test, unique_terms, top_k=5
            )
        logger.info(f"测试颜色名称: {color_name}, 真实颜色: {rgb}, 推荐颜色: {rgb_values}, 置信度: {scores}")

    return topk_results

def evaluate_trained_model(model_type="original", model_dir=None):
    """
    评估已训练的模型
    
    Args:
        model_type: 模型类型 ("original" 或 "pretrained_transformer")
        model_dir: 模型目录路径，如果为None则使用默认路径
    """
    logger.info("=== 评估已训练的模型 ===")
    
    # 确定模型目录
    if model_dir is None:
        if model_type == "pretrained_transformer":
            model_dir = "models-pretrained"
        else:
            model_dir = "models"
    
    logger.info(f"从目录加载模型: {model_dir}")
    
    # # 加载预处理数据
    # if not os.path.exists('models/preprocessed_data.pkl'):
    #     logger.info("错误: 找不到预处理数据文件 'models/preprocessed_data.pkl'")
    #     logger.info("请先运行训练流程生成数据")
    #     return
    
    # with open('models/preprocessed_data.pkl', 'rb') as f:
    #     data = pickle.load(f)
    #     rgb_values = data['rgb_values']
    #     encoded_labels = data['encoded_labels']
    #     unique_terms = data['unique_terms']
    #     unique_colors = data['unique_colors']

    # 1. 数据预处理
    rgb_values, encoded_labels, label_encoder, unique_terms, unique_colors = preprocess_data(
        csv_path='responses_cleaned_all.csv', min_count=10
    )

    term_color_mapping = load_xkcd_color_mapping()
    logger.info(f"term_color_mapping including {len(term_color_mapping)} terms")


    # 将每个rgb_value转换为hsl、lab、hcl、cmyk，并将所有格式拼接为16维特征，替换原有rgb_values
    # 生成16维新特征
    if use_16d_feature:
        new_features = []
        for rgb in rgb_values:
            feature = rgb_to_16d_feature(rgb)
            new_features.append(feature)
        rgb_values = np.array(new_features, dtype=np.float32)
    
    # 确保unique_terms是字符串列表
    if not isinstance(unique_terms, list):
        logger.info(f"警告: unique_terms类型为{type(unique_terms)}，尝试转换...")
        try:
            unique_terms = list(unique_terms)
        except Exception as e:
            logger.info(f"转换失败: {e}")
            return
    
    # 确保所有元素都是字符串
    unique_terms = [str(term) for term in unique_terms]
    
    logger.info(f"加载数据: {len(rgb_values)} 样本, {len(unique_terms)} 个颜色名称")
    logger.info(f"前5个颜色名称示例: {unique_terms[:5]}")
    logger.info(f"唯一RGB值: {len(unique_colors)} 个, 前5个唯一RGB值示例: {unique_colors[:5]}")
    logger.info(f"随机10个唯一RGB值示例: {unique_colors[np.random.randint(0, len(unique_colors), 10)]}")
    # 将unique_terms导出到CSV文件
    # unique_terms_df = pd.DataFrame({'term': unique_terms})
    # unique_terms_csv_path = os.path.join(model_dir, 'unique_terms.csv')
    # unique_terms_df.to_csv(unique_terms_csv_path, index=False, encoding='utf-8')
    # logger.info(f"已将unique_terms导出到: {unique_terms_csv_path}")
    
    # 划分训练集和测试集
    used_test_size = 10100 / len(rgb_values)
    logger.info(f"used_test_size: {used_test_size}")
    # 先全部分配给训练集，再手动划分10000个测试样本
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        rgb_values, encoded_labels, test_size=used_test_size, random_state=42
    )
    if len(X_test_full) > 10000:
        X_test = X_test_full[:10000]
        y_test = y_test_full[:10000]
        X_train = np.concatenate([X_train_full, X_test_full[10000:]], axis=0)
        y_train = np.concatenate([y_train_full, y_test_full[10000:]], axis=0)
    else:
        X_test = X_test_full
        y_test = y_test_full
        X_train = X_train_full
        y_train = y_train_full


    # # 按照颜色名称的频次从低到高排序，前10000为测试集，其余为训练集
    # # 统计每个颜色名称的出现次数
    # name_freq = Counter(encoded_labels)
    # # 为每个样本分配其对应的频次
    # sample_freq = np.array([name_freq[label] for label in encoded_labels])
    # # 获取频次从低到高的排序索引
    # sorted_indices = np.argsort(sample_freq)
    # # 前10000个为测试集，其余为训练集
    # test_indices = sorted_indices[:10000]
    # train_indices = sorted_indices[10000:]

    # X_test = rgb_values[test_indices]
    # y_test = encoded_labels[test_indices]
    # X_train = rgb_values[train_indices]
    # y_train = encoded_labels[train_indices]
    logger.info(f"训练集样本数: {len(X_train)}, 测试集样本数: {len(X_test)}")
    # export_test_data_to_csv(X_test, y_test, unique_terms, 'test_data_10000.csv')

    # 统计y_train中出现的颜色名称及其出现次数
    name_count_dict = {}
    for idx in y_train:
        name = unique_terms[idx]
        if name in name_count_dict:
            name_count_dict[name] += 1
        else:
            name_count_dict[name] = 1
    logger.info(f"训练集中出现的颜色名称数量: {len(name_count_dict)}")
    # 打印出现次数最多的前20个颜色名称及其出现次数
    top20 = sorted(name_count_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("出现次数最多的前10个颜色名称及其出现次数：")
    for name, count in top20:
        logger.info(f"  {name}: {count}")
    logger.info('-'*30)
    color_database = np.unique(X_train, axis=0)
    # 只保留y_train中出现过的颜色名称，unique_terms为这些唯一颜色名的列表
    name_database = list(set([unique_terms[idx] for idx in y_train]))
    logger.info(f"训练集样本数: {len(X_train)}， 包含 {len(color_database)} 个唯一RGB值， 包含 {len(name_database)} 个唯一颜色名称")
    logger.info('*'*60)
    name_count_dict = {}
    for idx in y_test:
        name = unique_terms[idx]
        if name in name_count_dict:
            name_count_dict[name] += 1
        else:
            name_count_dict[name] = 1
    logger.info(f"测试集中出现的颜色名称数量: {len(name_count_dict)}")
    # 打印出现次数最多的前20个颜色名称及其出现次数
    top20 = sorted(name_count_dict.items(), key=lambda x: x[1], reverse=True)[:20]
    logger.info("出现次数最多的前10个颜色名称及其出现次数：")
    for name, count in top20:
        logger.info(f"  {name}: {count}")
    logger.info('-'*30)
    color_database = np.unique(X_test, axis=0)
    # 只保留y_train中出现过的颜色名称，unique_terms为这些唯一颜色名的列表
    name_database = list(set([unique_terms[idx] for idx in y_test]))
    logger.info(f"测试集样本数: {len(X_test)}， 包含 {len(color_database)} 个唯一RGB值， 包含 {len(name_database)} 个唯一颜色名称")


    # 加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载预训练Transformer模型
    rgb_encoder = RGBEncoder(emb_dim=64, input_dim=rgb_values.shape[1]).to(device)
    name_encoder = NameEncoder(
        model_name='bert-base-uncased',
        emb_dim=64,
        use_pooling=False,  # 禁用注意力池化，使用CLS token或平均池化
        local_model_path="models-pretrained/bert-base-uncased"
    ).to(device)
    
    # 加载模型权重
    model_path = os.path.join(model_dir, f'model_best{suffix_str}.pt')
    logger.info(f"加载模型权重: {model_path}")
    if not os.path.exists(model_path):
        logger.info(f"错误: 找不到模型文件 {model_path}")
        return
        
    checkpoint = torch.load(model_path, map_location=device)
    rgb_encoder.load_state_dict(checkpoint['rgb_encoder'])
    name_encoder.load_state_dict(checkpoint['name_encoder'])
    
    # 初始化NCF模型（如果存在）
    ncf_model = None
    if 'ncf_model' in checkpoint:
        logger.info("检测到NCF模型权重，初始化NCF模型...")
        ncf_model = NCFModel(emb_dim=64).to(device)
        ncf_model.load_state_dict(checkpoint['ncf_model'])
        logger.info("NCF模型加载成功！")
    
    # 检查是否有RGB生成器权重
    rgb_generator = None
    if 'rgb_generator' in checkpoint:
        logger.info("🔍 检测到RGB生成器权重，初始化RGB生成器...")
        rgb_generator = RGBGenerator(
            input_dim=64,
            hidden_dims=rgb_generator_hidden_dims,
            output_dim=3
        ).to(device)
        rgb_generator.load_state_dict(checkpoint['rgb_generator'])
        logger.info("RGB生成器加载成功！")

    logger.info("模型加载成功！")
    
    # 测试推理
    logger.info("\n=== 测试推理 ===")
    
    # test_compute_name2color_accuracy(rgb_encoder, name_encoder, ncf_model, unique_colors, top_k=5)
    evaluation_results = evaluate_model(rgb_encoder, name_encoder, ncf_model, X_test, y_test, unique_terms, unique_colors, rgb_generator)
   
    # 打印评估结果摘要
    logger.info(f"\n{'='*60}")
    logger.info("=== 评估结果摘要 ===")
    
    if 'topk_accuracy' in evaluation_results:
        logger.info("color2name Top-K 准确率结果:")
        for key, value in evaluation_results['topk_accuracy'].items():
            if 'accuracy' in key:
                logger.info(f"  {key}: {value:.4f}")
    
    logger.info(f"{'='*20}")
    if 'name2color_accuracy' in evaluation_results:
        logger.info(f"name2color 推荐: {evaluation_results['name2color_accuracy']['cielab_distance_mean']:.4f}")
    
    logger.info(f"{'='*20}")
    if 'text2color_accuracy' in evaluation_results:
        logger.info(f"name2color 生成:{evaluation_results['text2color_accuracy']['cielab_distance_mean']:.4f}")
    
def export_test_data_to_csv(X_test, y_test, unique_terms, output_path='test_data.csv'):
    """
    将测试数据导出为CSV格式
    
    Args:
        X_test: 测试集特征 (16维或3维)
        y_test: 测试集标签索引
        unique_terms: 颜色名称列表
        output_path: 输出CSV文件路径
    """
    logger.info(f"=== 导出测试数据到CSV ===")
    logger.info(f"测试集样本数: {len(X_test)}")
    logger.info(f"颜色名称数量: {len(unique_terms)}")
    
    # 确保unique_terms是列表格式
    if isinstance(unique_terms, np.ndarray):
        unique_terms_list = unique_terms.tolist()
    else:
        unique_terms_list = list(unique_terms)
    
    # 准备数据列表
    data_list = []
    
    for i, (features, label_idx) in enumerate(zip(X_test, y_test)):
        # 获取RGB值（前3维）
        if len(features) >= 3:
            r, g, b = features[:3]
        else:
            r, g, b = 0.0, 0.0, 0.0
            logger.info(f"警告: 样本 {i} 特征维度不足: {len(features)}")
        
        # 获取颜色名称
        if 0 <= label_idx < len(unique_terms_list):
            term = unique_terms_list[label_idx]
        else:
            term = f"unknown_{label_idx}"
            logger.info(f"警告: 样本 {i} 标签索引超出范围: {label_idx}")
        
        lab = rgb_to_lab([r, g, b])
        # 添加到数据列表
        data_list.append({
            'r': float(r),
            'g': float(g), 
            'b': float(b),
            'lab_l': float(lab[0]),
            'lab_a': float(lab[1]),
            'lab_b': float(lab[2]),
            'term': str(term)
        })
        
        # 显示进度
        if (i + 1) % 1000 == 0:
            logger.info(f"  处理进度: {i + 1}/{len(X_test)}")
    
    # 创建DataFrame
    df = pd.DataFrame(data_list)
    
    # 显示数据预览
    logger.info(f"\n数据预览:")
    logger.info(f"数据形状: {df.shape}")
    logger.info(f"列名: {list(df.columns)}")
    logger.info(f"前5行数据:")
    logger.info(df.head())
    
    # 显示数据类型
    logger.info(f"\n数据类型:")
    logger.info(df.dtypes)
    
    # 显示基本统计信息
    logger.info(f"\nRGB值统计:")
    logger.info(f"R值范围: [{df['r'].min():.3f}, {df['r'].max():.3f}]")
    logger.info(f"G值范围: [{df['g'].min():.3f}, {df['g'].max():.3f}]")
    logger.info(f"B值范围: [{df['b'].min():.3f}, {df['b'].max():.3f}]")
    
    # 显示颜色名称分布
    logger.info(f"\n颜色名称分布 (前10个):")
    term_counts = df['term'].value_counts()
    logger.info(term_counts.head(10))
    
    # 保存到CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    logger.info(f"\n✅ 数据已成功导出到: {output_path}")
    logger.info(f"文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
    
    return df

def test_compute_name2color_accuracy(rgb_encoder, name_encoder, ncf_model, unique_colors,
                               top_k=5):
    """
    测试模型对于term_color_mapping中的名称和颜色是否能匹配到相似的颜色
    
    Args:
        rgb_encoder: RGB编码器
        name_encoder: 名称编码器
        ncf_model: NCF模型（可选）
        unique_colors: 唯一颜色值列表
        term_color_mapping: 颜色映射字典
        top_k: 推荐数量
        test_sample_size: 测试样本数量
    
    Returns:
        results: 包含准确率统计的字典
    """
    logger.info(f"=== 测试名称到颜色匹配准确率 ===")
    logger.info(f"测试目标: term_color_mapping中的名称和颜色匹配")
    term_color_mapping = load_xkcd_color_mapping()
    logger.info(f"term_color_mapping including {len(term_color_mapping)} terms")
    
    if term_color_mapping is None or len(term_color_mapping) == 0:
        logger.info("❌ 错误: term_color_mapping为空，无法进行测试")
        return None
    
    # 随机生成len(term_color_mapping)个RGB，计算平均CIELAB距离
    n_samples = len(term_color_mapping)
    logger.info(f"\n=== 随机生成 {n_samples} 个RGB，计算平均CIELAB距离 ===")
    # 随机生成RGB值（0-1之间）
    random_rgbs = np.random.rand(n_samples, 3).astype(np.float32)

    # 计算每个随机RGB与其最近的term_color_mapping颜色的CIELAB距离
    mapping_rgbs = np.array([v['rgb'] for v in term_color_mapping.values()], dtype=np.float32)
    mapping_rgbs = mapping_rgbs / 255.0  # 保证归一化

    cielab_distances = []
    for i in range(n_samples):
        random_lab = rgb_to_lab(random_rgbs[i])
        mapping_lab = rgb_to_lab(mapping_rgbs[i])
        cielab_dist = np.sqrt(np.sum((np.array(random_lab) - np.array(mapping_lab))**2))
        cielab_distances.append(cielab_dist)

    mean_cielab_dist = np.mean(cielab_distances)
    std_cielab_dist = np.std(cielab_distances)
    logger.info(f"随机RGB与最近term_color_mapping颜色的CIELAB距离: 平均={mean_cielab_dist:.4f}, 标准差={std_cielab_dist:.4f}")
    

    rgb_encoder.eval()
    name_encoder.eval()
    if ncf_model is not None:
        ncf_model.eval()
    
    device = next(rgb_encoder.parameters()).device
    logger.info(f"使用设备: {device}")
    
    # 从term_color_mapping中随机选择测试样本
    import random
    random.seed(42)  # 确保可重现性
    
    # 获取可用的颜色名称
    test_terms = list(term_color_mapping.keys())
    logger.info(f"term_color_mapping中包含 {len(test_terms)} 个颜色名称")
    
    # 初始化统计变量
    cielab_distances = []
    rgb_distances = []
    successful_matches = 0
    
    logger.info(f"开始评估 {len(test_terms)} 个颜色名称...")
    
    logger.info("🔄 预计算所有RGB的embedding...")
    start_time = time.time()
    
    # 将unique_colors转换为numpy数组并移到设备上
    if isinstance(unique_colors[0], tuple):
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    else:
        unique_colors_array = np.array(unique_colors, dtype=np.float32)
    
    # 分批次计算RGB embedding
    rgb_batch_size = 10000
    all_rgb_embeddings = []
    
    for i in range(0, len(unique_colors_array), rgb_batch_size):
        batch_colors = unique_colors_array[i:i+rgb_batch_size]
        batch_tensor = torch.tensor(batch_colors, dtype=torch.float32, device=device)
        
        with torch.no_grad():
            batch_embeddings = rgb_encoder(batch_tensor)
            all_rgb_embeddings.append(batch_embeddings.cpu())
        
        if (i // rgb_batch_size + 1) % 10 == 0:
            logger.info(f"  RGB embedding进度: {i + rgb_batch_size}/{len(unique_colors_array)}")
    
    all_rgb_embeddings = torch.cat(all_rgb_embeddings, dim=0).to(device)
    logger.info(f"✅ RGB embedding预计算完成，形状: {all_rgb_embeddings.shape}")
    logger.info(f"⏱️ RGB embedding预计算耗时: {time.time() - start_time:.2f}秒")
    
    # 🚀 优化2: 预计算测试颜色名称的embedding
    logger.info("🔄 预计算测试颜色名称的embedding...")
    start_time = time.time()
    
    # 分批次计算名称embedding
    name_batch_size = 100
    all_name_embeddings = []
    
    for i in range(0, len(test_terms), name_batch_size):
        batch_names = test_terms[i:i+name_batch_size]
        with torch.no_grad():
            batch_embeddings = name_encoder(batch_names)
            all_name_embeddings.append(batch_embeddings.cpu())
        
        if (i // name_batch_size + 1) % 10 == 0:
            logger.info(f"  名称embedding进度: {i + name_batch_size}/{len(test_terms)}")
    
    all_name_embeddings = torch.cat(all_name_embeddings, dim=0).to(device)
    logger.info(f"✅ 名称embedding预计算完成，形状: {all_name_embeddings.shape}")
    logger.info(f"⏱️ 名称embedding预计算耗时: {time.time() - start_time:.2f}秒")
    
    # 🚀 优化3: 开始评估
    logger.info("🔄 开始评估名称到颜色匹配...")
    start_time = time.time()
    
    # 根据显存动态调整批次大小
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024**3
        if gpu_memory >= 24:
            test_batch_size = 100
        elif gpu_memory >= 16:
            test_batch_size = 50
        else:
            test_batch_size = 20
    else:
        test_batch_size = 20
    
    logger.info(f"💾 GPU显存: {gpu_memory:.1f}GB, 测试批次大小: {test_batch_size}")
    
    n_test_batches = (len(test_terms) + test_batch_size - 1) // test_batch_size
    
    for batch_idx in range(n_test_batches):
        start_idx = batch_idx * test_batch_size
        end_idx = min(start_idx + test_batch_size, len(test_terms))
        
        batch_terms = test_terms[start_idx:end_idx]
        batch_name_embs = all_name_embeddings[start_idx:end_idx]
        
        with torch.no_grad():
            # 计算当前批次与所有RGB的相似度
            if ncf_model is not None:
                # 使用NCF模型计算相似度
                batch_scores = []
                for i in range(len(batch_name_embs)):
                    name_emb = batch_name_embs[i].unsqueeze(0).repeat(len(all_rgb_embeddings), 1)
                    scores = ncf_model(all_rgb_embeddings, name_emb)
                    batch_scores.append(scores.squeeze().cpu().numpy())
                batch_similarity = np.stack(batch_scores, axis=0)
            else:
                # 使用余弦相似度
                batch_similarity = torch.matmul(batch_name_embs, all_rgb_embeddings.T).cpu().numpy()
            
            # 为每个颜色名称找到top-k最相似的RGB值
            for i, term in enumerate(batch_terms):
                # 获取真实RGB值
                if term in term_color_mapping:
                    true_rgb = term_color_mapping[term]['rgb']
                    # 转换为0-1范围
                    true_rgb_01 = [c/255.0 for c in true_rgb]
                else:
                    logger.info(f"⚠️ 警告: 颜色名称 '{term}' 不在term_color_mapping中")
                    continue
                
                # 获取top-k推荐
                similarity_scores = batch_similarity[i]
                top_k_indices = np.argsort(similarity_scores)[::-1][:top_k]
                
                # 获取第一个推荐（最高分数）
                pred_rgb_idx = top_k_indices[0]
                pred_rgb = unique_colors_array[pred_rgb_idx]
                
                # 计算RGB欧几里得距离
                rgb_dist = np.sqrt(np.sum((np.array(pred_rgb[:3]) - np.array(true_rgb_01))**2))
                rgb_distances.append(rgb_dist)
                
                # 计算CIELAB距离
                try:
                    true_lab = rgb_to_lab(true_rgb_01)
                    pred_lab = rgb_to_lab(pred_rgb[:3])
                    cielab_dist = np.sqrt(np.sum((np.array(pred_lab) - np.array(true_lab))**2))
                    cielab_distances.append(cielab_dist)
                    
                    # 判断是否匹配成功（CIELAB距离小于阈值）
                    if cielab_dist < 5:  # 阈值可调整
                        successful_matches += 1
                        
                except Exception as e:
                    logger.info(f"⚠️ 样本 {term} CIELAB转换失败: {e}")
                    cielab_distances.append(rgb_dist)  # 使用RGB距离作为fallback
                
                # 显示前几个样本的详细信息
                if batch_idx == 0 and i < 5:
                    logger.info(f"  样本 {term}:")
                    logger.info(f"    真实RGB: {true_rgb_01}")
                    logger.info(f"    预测RGB: {pred_rgb[:3]}")
                    logger.info(f"    RGB距离: {rgb_dist:.4f}")
                    if 'cielab_dist' in locals():
                        logger.info(f"    CIELAB距离: {cielab_dist:.4f}")
                    logger.info(f"    相似度分数: {similarity_scores[pred_rgb_idx]:.4f}")
        
        # 显示进度
        if (batch_idx + 1) % 5 == 0:
            logger.info(f"  测试批次进度: {batch_idx + 1}/{n_test_batches}")
    
    logger.info(f"✅ 评估完成，耗时: {time.time() - start_time:.2f}秒")
    
    # 计算统计结果
    valid_rgb_dists = [d for d in rgb_distances if d != float('inf') and not np.isnan(d)]
    valid_cielab_dists = [d for d in cielab_distances if d != float('inf') and not np.isnan(d)]
    
    results = {
        'total_samples': len(test_terms),
        'successful_matches': successful_matches,
        'success_rate': successful_matches / len(test_terms) if len(test_terms) > 0 else 0,
        'rgb_distance_mean': np.mean(valid_rgb_dists),
        'rgb_distance_std': np.std(valid_rgb_dists),
        'rgb_distance_min': np.min(valid_rgb_dists),
        'rgb_distance_max': np.max(valid_rgb_dists),
        'cielab_distance_mean': np.mean(valid_cielab_dists),
        'cielab_distance_std': np.std(valid_cielab_dists),
        'cielab_distance_min': np.min(valid_cielab_dists),
        'cielab_distance_max': np.max(valid_cielab_dists)
    }
    
    # 打印结果
    logger.info(f"\n=== 名称到颜色匹配测试结果 ===")
    logger.info(f"总测试样本数: {results['total_samples']}")
    logger.info(f"成功匹配数: {results['successful_matches']}")
    logger.info(f"成功率: {results['success_rate']:.2%}")
    logger.info(f"RGB距离 - 平均: {results['rgb_distance_mean']:.4f}, 标准差: {results['rgb_distance_std']:.4f}")
    logger.info(f"RGB距离 - 最小: {results['rgb_distance_min']:.4f}, 最大: {results['rgb_distance_max']:.4f}")
    logger.info(f"CIELAB距离 - 平均: {results['cielab_distance_mean']:.4f}, 标准差: {results['cielab_distance_std']:.4f}")
    logger.info(f"CIELAB距离 - 最小: {results['cielab_distance_min']:.4f}, 最大: {results['cielab_distance_max']:.4f}")
    
    # 🚀 清理GPU内存
    logger.info("🧹 清理GPU内存...")
    del all_rgb_embeddings, all_name_embeddings
    torch.cuda.empty_cache()

    return results

if __name__ == "__main__":
    # 检查命令行参数
    
    # logger.info(rgb_to_lab([0.6745098039215687, 0.7607843137254902, 0.8509803921568627]))
    if len(sys.argv) > 1 and sys.argv[1] == "evaluate":
        logger.info("运行评估流程...")
        # 评估模式
        model_type = "pretrained_transformer"
        
        evaluate_trained_model(model_type)
    else:
        logger.info("运行训练流程...")
        # 训练模式
        main()
    
    logger.info(f"ending time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))}")