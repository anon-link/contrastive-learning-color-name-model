 # -*- coding: utf-8 -*-
from flask import Flask, render_template, request, jsonify
import torch
import numpy as np
import pickle
import os
import sys
import csv
import time
import logging

# 添加项目根目录到路径
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 仅在反序列化阶段做兼容映射，避免运行期导入冲突
class NumpyCompatUnpickler(pickle.Unpickler):
	def find_class(self, module, name):
		if module and module.startswith('numpy._core'):
			module = module.replace('numpy._core', 'numpy.core', 1)
		return super().find_class(module, name)

# 导入模型相关函数
from color_utils import rgb_to_16d_feature, rgb_to_lab
# 从模型文件中导入必要的类
from color_name_model import RGBEncoder, NameEncoder, NCFModel, RGBGenerator

app = Flask(__name__)

# 全局变量
rgb_encoder = None
name_encoder = None
ncf_model = None
rgb_generator = None
unique_terms = None
unique_colors = None
use_16d_feature = True

# 预计算的embeddings（用于加速推荐）
precomputed_name_embeddings = None
precomputed_rgb_embeddings = None

# 防止重复加载的标志
model_loaded = False

# Embedding 缓存文件路径
EMB_CACHE_DIR = 'models'
NAME_EMB_FILE = os.path.join(EMB_CACHE_DIR, 'name_embeddings.pt')
RGB_EMB_FILE = os.path.join(EMB_CACHE_DIR, 'rgb_embeddings.pt')

def calibrate_scores(raw_scores: np.ndarray, gamma: float = 2.5, alpha: float = 0.88) -> np.ndarray:
    """对top-k原始分数进行校准，提升可分性：
    - 若分数范围极小（几乎相等），按名次几何衰减：p_i ∝ alpha^(rank-1)
    - 否则：min-max归一化到[0,1]后做幂次拉伸，再归一化为概率
    返回：长度与raw_scores相同的概率数组（和为1）。
    """
    raw_scores = np.asarray(raw_scores, dtype=np.float64)
    if raw_scores.size == 0:
        return raw_scores
    s_min = float(np.min(raw_scores))
    s_max = float(np.max(raw_scores))
    if s_max - s_min < 1e-8:
        # 分数几乎一致，使用名次几何衰减
        ranks = np.arange(1, len(raw_scores) + 1)
        # raw_scores 应按从高到低传入；若不确定，先排序，再还原顺序
        # 这里假设传入顺序即为排序后顺序
        weights = alpha ** (ranks - 1)
        probs = weights / np.sum(weights)
        return probs.astype(np.float64)
    # min-max归一化
    norm = (raw_scores - s_min) / (s_max - s_min)
    # 幂次拉伸（gamma>1 提高头部占比）
    stretched = np.power(norm, gamma)
    total = np.sum(stretched)
    if total <= 0:
        return np.full_like(stretched, 1.0 / len(stretched))
    return (stretched / total).astype(np.float64)

def load_cached_embeddings(device):
    """尝试从磁盘加载已缓存的embeddings。成功返回True。"""
    global precomputed_name_embeddings, precomputed_rgb_embeddings, unique_terms, unique_colors
    try:
        if os.path.exists(NAME_EMB_FILE) and os.path.exists(RGB_EMB_FILE):
            # 检查文件大小，如果太大则跳过缓存加载
            name_file_size = os.path.getsize(NAME_EMB_FILE) / 1024 / 1024  # MB
            rgb_file_size = os.path.getsize(RGB_EMB_FILE) / 1024 / 1024   # MB
            total_size = name_file_size + rgb_file_size
            
            if total_size > 1000:  # 如果缓存文件总大小超过1000MB
                print(f"缓存文件太大 ({total_size:.1f}MB)，跳过缓存加载，使用实时计算模式")
                return False
            
            print(f"尝试加载缓存文件 (名称: {name_file_size:.1f}MB, RGB: {rgb_file_size:.1f}MB)...")
            
            name_emb = torch.load(NAME_EMB_FILE, map_location='cpu')
            rgb_emb = torch.load(RGB_EMB_FILE, map_location='cpu')

            # 基本一致性校验：行数匹配样本数量
            if name_emb.size(0) == len(unique_terms) and rgb_emb.size(0) == len(unique_colors):
                precomputed_name_embeddings = name_emb.to(device)
                precomputed_rgb_embeddings = rgb_emb.to(device)
                print(f"已从缓存加载名称embeddings: {precomputed_name_embeddings.shape}")
                print(f"已从缓存加载RGB embeddings: {precomputed_rgb_embeddings.shape}")
                return True
            else:
                print("缓存的embeddings与当前数据集大小不匹配，跳过使用缓存")
                return False
        else:
            print("未找到embeddings缓存文件，将进行预计算")
            return False
    except Exception as e:
        print(f"加载embeddings缓存失败: {e}")
        print("将使用实时计算模式")
        return False

def save_cached_embeddings():
    """将当前预计算的embeddings保存到磁盘。"""
    global precomputed_name_embeddings, precomputed_rgb_embeddings
    try:
        if precomputed_name_embeddings is None or precomputed_rgb_embeddings is None:
            return False
        os.makedirs(EMB_CACHE_DIR, exist_ok=True)
        # 保存到CPU以减小文件与避免GPU依赖
        torch.save(precomputed_name_embeddings.detach().cpu(), NAME_EMB_FILE)
        torch.save(precomputed_rgb_embeddings.detach().cpu(), RGB_EMB_FILE)
        print(f"已将名称embeddings保存到: {NAME_EMB_FILE}")
        print(f"已将RGB embeddings保存到: {RGB_EMB_FILE}")
        return True
    except Exception as e:
        print(f"保存embeddings缓存失败: {e}")
        return False

def precompute_embeddings():
    """预计算所有颜色名称和RGB的embeddings"""
    global precomputed_name_embeddings, precomputed_rgb_embeddings, name_encoder, rgb_encoder, unique_terms, unique_colors
    
    if name_encoder is None or rgb_encoder is None or unique_terms is None or unique_colors is None:
        print("Models or data not loaded, skipping precomputation")
        return False
    
    # 检查数据量，如果太大则跳过预计算
    estimated_memory_mb = (len(unique_terms) * 128 * 4 + len(unique_colors) * 64 * 4) / 1024 / 1024
    if estimated_memory_mb > 1000:  # 如果估计需要超过1000MB内存
        print(f"数据量太大 (估计需要 {estimated_memory_mb:.1f}MB 内存)，跳过预计算，使用实时计算模式")
        return False
    
    try:
        device = next(name_encoder.parameters()).device
        print("开始预计算embeddings...")
        
        # 预计算颜色名称embeddings
        print("预计算颜色名称embeddings...")
        name_batch_size = 1000
        all_name_embeddings = []
        
        for i in range(0, len(unique_terms), name_batch_size):
            batch_names = unique_terms[i:i+name_batch_size]
            if isinstance(batch_names, np.ndarray):
                batch_names = batch_names.tolist()
            
            with torch.no_grad():
                batch_embeddings = name_encoder.encode_batch(batch_names)
                all_name_embeddings.append(batch_embeddings.cpu())
        
        precomputed_name_embeddings = torch.cat(all_name_embeddings, dim=0).to(device)
        print(f"颜色名称embeddings预计算完成: {precomputed_name_embeddings.shape}")
        
        # 预计算RGB embeddings
        print("预计算RGB embeddings...")
        if isinstance(unique_colors[0], tuple):
            unique_colors_array = np.array(unique_colors, dtype=np.float32)
        else:
            unique_colors_array = np.array(unique_colors, dtype=np.float32)
        
        rgb_batch_size = 100000
        all_rgb_embeddings = []
        
        for i in range(0, len(unique_colors_array), rgb_batch_size):
            start_idx = i
            end_idx = min(start_idx + rgb_batch_size, len(unique_colors_array))
            batch_colors = unique_colors_array[start_idx:end_idx]
            
            # 根据use_16d_feature设置处理RGB特征
            batch_features = []
            for rgb in batch_colors:
                if use_16d_feature:
                    if len(rgb) == 3:
                        feature = rgb_to_16d_feature(rgb)
                    elif len(rgb) == 16:
                        feature = rgb
                    else:
                        feature = list(rgb[:3]) + [0.0] * 13
                else:
                    if len(rgb) >= 3:
                        feature = rgb[:3]
                    else:
                        feature = list(rgb) + [0.0] * (3 - len(rgb))
                
                batch_features.append(feature)
            
            batch_features = np.array(batch_features, dtype=np.float32)
            batch_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
            
            with torch.no_grad():
                batch_embeddings = rgb_encoder(batch_tensor)
                all_rgb_embeddings.append(batch_embeddings.cpu())
        
        precomputed_rgb_embeddings = torch.cat(all_rgb_embeddings, dim=0).to(device)
        print(f"RGB embeddings预计算完成: {precomputed_rgb_embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"预计算embeddings失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def load_model(emb_dim=64, device=None):
    """加载训练好的模型"""
    global rgb_encoder, name_encoder, ncf_model, rgb_generator, unique_terms, unique_colors, precomputed_name_embeddings, precomputed_rgb_embeddings, model_loaded
    
    # 如果已经加载过，直接返回
    if model_loaded:
        print("Model already loaded, skipping...")
        return True
    
    try:
        # 加载预处理数据（使用兼容的 Unpickler 以适配 numpy._core -> numpy.core）
        with open('models/preprocessed_data.pkl', 'rb') as f:
            data = NumpyCompatUnpickler(f).load()
            unique_terms = data['unique_terms']
            unique_colors = data['unique_colors']
            
            # Debug info: check unique_colors raw data
            print(f"unique_colors type: {type(unique_colors)}")
            print(f"unique_colors length: {len(unique_colors)}")
            if len(unique_colors) > 0:
                # Output random colors
                import random
                random_indices = random.sample(range(len(unique_colors)), min(10, len(unique_colors)))
                random_colors = [unique_colors[i] for i in random_indices]
                print(f"Randomly selected 10 colors: {random_colors}")
        
        # 加载模型
        model_path = 'models/model_best_0909_test_False_ncf_0.01_generator_1_epoch_60_device_0_name_1_ab_0.0_high_False_freezed_True.pt'
        checkpoint = torch.load(model_path, map_location='cpu')
        
        
        # 初始化模型
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        rgb_encoder = RGBEncoder(emb_dim=emb_dim, input_dim=16)  
        name_encoder = NameEncoder(model_name='bert-base-uncased',
            emb_dim=emb_dim,
            freeze_pretrained=True,  # 冻结预训练参数
            use_pooling=False,  # 禁用注意力池化，使用CLS token或平均池化
            local_model_path = "models-pretrained/bert-base-uncased")
        ncf_model = NCFModel(emb_dim=emb_dim)
        rgb_generator = RGBGenerator(input_dim=emb_dim, hidden_dims=[])
        
        # 加载权重
        if 'rgb_encoder' in checkpoint:
            rgb_encoder.load_state_dict(checkpoint['rgb_encoder'])
            print("已加载RGB编码器权重")
        
        if 'name_encoder' in checkpoint:
            # 兼容 Transformers 版本差异：缺失 position_ids 等非训练参数时不报错
            try:
                # 若可访问到预训练模型的 embeddings，补注册默认 position_ids
                pretrained = getattr(name_encoder, 'pretrained_model', None)
                if pretrained is not None and hasattr(pretrained, 'embeddings'):
                    emb = pretrained.embeddings
                    if not hasattr(emb, 'position_ids') or getattr(emb, 'position_ids') is None:
                        try:
                            max_len = emb.position_embeddings.num_embeddings
                            pos_ids = torch.arange(max_len).unsqueeze(0)
                            emb.register_buffer('position_ids', pos_ids)
                            print(f"已注册默认 position_ids，长度: {max_len}")
                        except Exception:
                            pass
            except Exception:
                pass

            load_info = name_encoder.load_state_dict(checkpoint['name_encoder'], strict=False)
            try:
                missing = getattr(load_info, 'missing_keys', [])
                unexpected = getattr(load_info, 'unexpected_keys', [])
                if missing:
                    print(f"NameEncoder 缺失权重键(已忽略): {missing}")
                if unexpected:
                    print(f"NameEncoder 额外权重键(已忽略): {unexpected}")
            except Exception:
                pass
            print("已加载名称编码器权重（strict=False）")
        
        if 'ncf_model' in checkpoint:
            ncf_model.load_state_dict(checkpoint['ncf_model'])
            print("已加载NCF模型权重")
        
        if 'rgb_generator' in checkpoint:
            rgb_generator.load_state_dict(checkpoint['rgb_generator'])
            print("已加载RGB生成器权重")
        else:
            rgb_generator = None
            print("未找到RGB生成器权重")
        
        # 将模型移到设备上
        rgb_encoder.to(device)
        name_encoder.to(device)
        ncf_model.to(device)
        if rgb_generator is not None:
            rgb_generator.to(device)
        
        # 设置为评估模式
        rgb_encoder.eval()
        name_encoder.eval()
        ncf_model.eval()
        if rgb_generator is not None:
            rgb_generator.eval()
        
        print(f"Model loaded successfully! Vocabulary size: {len(unique_terms)}")
        print(f"Color database size: {len(unique_colors)}")
        
        # 优先尝试从缓存加载embeddings
        device = next(ncf_model.parameters()).device
        if load_cached_embeddings(device):
            print("已使用缓存的embeddings")
        else:
            # 预计算embeddings以加速推荐
            if precompute_embeddings():
                print("预计算完成！开始保存到缓存...")
                save_cached_embeddings()
            else:
                print("预计算失败，将使用实时计算模式")
        
        # 标记模型已加载
        model_loaded = True
        return True
        
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return False

def recommend_color_names(rgb_input, top_k=10):
    """为给定的RGB值推荐颜色名称（基于NCF模型，使用预计算embeddings加速）"""
    global rgb_encoder, name_encoder, ncf_model, unique_terms, precomputed_name_embeddings
    
    if rgb_encoder is None or ncf_model is None:
        return [], []
    
    try:
        device = next(ncf_model.parameters()).device
        
        # 将RGB转换为16维特征
        if use_16d_feature:
            feature_16d = rgb_to_16d_feature(rgb_input)
            feature_tensor = torch.tensor(feature_16d, dtype=torch.float32).to(device)
        else:
            feature_tensor = torch.tensor(rgb_input, dtype=torch.float32).to(device)
        
        ncf_model.eval()
        rgb_encoder.eval()
        
        with torch.no_grad():
            # 使用RGB编码器获取RGB embedding
            z_rgb = rgb_encoder(feature_tensor.unsqueeze(0))
            
            if precomputed_name_embeddings is not None:
                # 使用预计算的名称embeddings（快速模式）
                print("使用预计算的名称embeddings进行快速推荐")
                z_rgb_expanded = z_rgb.repeat(len(precomputed_name_embeddings), 1)
                all_scores = ncf_model(z_rgb_expanded, precomputed_name_embeddings)
                all_scores = all_scores.squeeze().cpu().numpy()
            else:
                # 回退到实时计算模式
                print("使用实时计算模式（预计算不可用）")
                all_scores = []
                batch_size = 1000
                
                for i in range(0, len(unique_terms), batch_size):
                    batch_names = unique_terms[i:i+batch_size]
                    if isinstance(batch_names, np.ndarray):
                        batch_names = batch_names.tolist()
                    
                    # 使用名称编码器获取名称embedding
                    z_names = name_encoder.encode_batch(batch_names)
                    
                    # 使用NCF模型计算相似度分数
                    z_rgb_expanded = z_rgb.repeat(len(batch_names), 1)
                    batch_scores = ncf_model(z_rgb_expanded, z_names)
                    all_scores.extend(batch_scores.squeeze().cpu().numpy())
                
                all_scores = np.array(all_scores)
            
            # 转换为numpy数组并排序
            top_indices = np.argsort(all_scores)[::-1][:top_k]
            
            # 确保unique_terms是列表格式
            if isinstance(unique_terms, np.ndarray):
                unique_terms_list = unique_terms.tolist()
            else:
                unique_terms_list = list(unique_terms)
            
            names = [unique_terms_list[i] for i in top_indices]
            raw_scores = all_scores[top_indices]
            # 分数校准，增强区分度
            scores = calibrate_scores(raw_scores, gamma=2, alpha=0.92)
        
        return names, scores
        
    except Exception as e:
        print(f"Error recommending color names: {e}")
        import traceback
        traceback.print_exc()
        return [], []

def generate_quick_color(color_name):
    """快速生成一个颜色（使用RGBGenerator）"""
    global name_encoder, rgb_generator
    
    if name_encoder is None or rgb_generator is None:
        return None
    
    try:
        device = next(rgb_generator.parameters()).device
        
        name_encoder.eval()
        rgb_generator.eval()
        
        with torch.no_grad():
            # 使用名称编码器获取颜色名称的embedding
            z_name = name_encoder.encode_batch([color_name])
            
            # 快速生成一个颜色
            generated_rgb = rgb_generator(z_name)
            generated_rgb = generated_rgb.squeeze().cpu().numpy()
            # 确保RGB值在[0,1]范围内
            generated_rgb = np.clip(generated_rgb, 0, 1)
            print(f"RGBGenerator quick color generation: {generated_rgb}")
            
            return generated_rgb
            
    except Exception as e:
        print(f"RGBGenerator generation failed: {e}")
        return None

def recommend_rgb_colors(color_name, top_k=10):
    """为给定的颜色名称推荐RGB值（基于NCF模型，使用预计算embeddings加速）"""
    global rgb_encoder, name_encoder, ncf_model, unique_colors, precomputed_rgb_embeddings
    
    if rgb_encoder is None or ncf_model is None or unique_colors is None:
        return [], []
    
    try:
        device = next(ncf_model.parameters()).device
        
        ncf_model.eval()
        name_encoder.eval()
        
        with torch.no_grad():
            # 使用名称编码器获取颜色名称的embedding
            z_name = name_encoder.encode_batch([color_name])
            
            # 将unique_colors转换为numpy数组
            if isinstance(unique_colors[0], tuple):
                unique_colors_array = np.array(unique_colors, dtype=np.float32)
            else:
                unique_colors_array = np.array(unique_colors, dtype=np.float32)
            
            if precomputed_rgb_embeddings is not None:
                # 使用预计算的RGB embeddings（快速模式）
                print("使用预计算的RGB embeddings进行快速推荐")
                all_rgb_embeddings = precomputed_rgb_embeddings
            else:
                # 回退到实时计算模式
                print("使用实时计算模式（预计算不可用）")
                rgb_batch_size = 100000
                all_rgb_embeddings = []
                
                for i in range(0, len(unique_colors_array), rgb_batch_size):
                    start_idx = i
                    end_idx = min(start_idx + rgb_batch_size, len(unique_colors_array))
                    batch_colors = unique_colors_array[start_idx:end_idx]
                    
                    # 根据use_16d_feature设置处理RGB特征
                    batch_features = []
                    for rgb in batch_colors:
                        if use_16d_feature:
                            if len(rgb) == 3:
                                feature = rgb_to_16d_feature(rgb)
                            elif len(rgb) == 16:
                                feature = rgb
                            else:
                                print(f"警告: 意外的RGB维度 {len(rgb)}，使用简单填充")
                                feature = list(rgb[:3]) + [0.0] * 13
                        else:
                            if len(rgb) >= 3:
                                feature = rgb[:3]
                            else:
                                print(f"警告: RGB维度不足 {len(rgb)}，使用零填充")
                                feature = list(rgb) + [0.0] * (3 - len(rgb))
                        
                        batch_features.append(feature)
                    
                    batch_features = np.array(batch_features, dtype=np.float32)
                    batch_tensor = torch.tensor(batch_features, dtype=torch.float32, device=device)
                    
                    batch_embeddings = rgb_encoder(batch_tensor)
                    all_rgb_embeddings.append(batch_embeddings.cpu())
                
                all_rgb_embeddings = torch.cat(all_rgb_embeddings, dim=0).to(device)
            
            # 使用分块计算避免大矩阵内存爆炸
            rgb_chunk_size = 100000  # 每次处理10万个RGB
            n_rgb_chunks = (len(unique_colors_array) + rgb_chunk_size - 1) // rgb_chunk_size
            
            # 维护全局top-k结果（增加候选数量以便去重）
            max_candidates = min(1000, len(unique_colors_array))  # 最多考虑1000个候选
            global_top_k_values = torch.full((max_candidates,), float('-inf'), device=device)
            global_top_k_indices = torch.zeros((max_candidates,), dtype=torch.long, device=device)
            
            # 分块处理RGB，避免大矩阵
            for rgb_chunk_idx in range(n_rgb_chunks):
                rgb_start = rgb_chunk_idx * rgb_chunk_size
                rgb_end = min(rgb_start + rgb_chunk_size, len(unique_colors_array))
                
                # 获取当前RGB块的embedding
                rgb_chunk_embeddings = all_rgb_embeddings[rgb_start:rgb_end]
                
                # 使用NCF模型计算相似度
                z_name_expanded = z_name.repeat(len(rgb_chunk_embeddings), 1)
                chunk_scores = ncf_model(rgb_chunk_embeddings, z_name_expanded)
                chunk_similarity = chunk_scores.squeeze()
                
                # 更新全局top-k结果
                # 获取当前chunk中的top-k
                chunk_values, chunk_indices = torch.topk(chunk_similarity, k=min(max_candidates, chunk_similarity.size(0)))
                
                # 与全局top-k比较，保留最大的
                combined_values = torch.cat([global_top_k_values, chunk_values])
                combined_indices = torch.cat([global_top_k_indices, chunk_indices + rgb_start])
                
                # 找到全局top-k
                global_top_k_values, global_top_k_indices = torch.topk(combined_values, k=max_candidates)
                global_top_k_indices = combined_indices[global_top_k_indices]
                
                # 清理当前chunk的显存
                del chunk_similarity, rgb_chunk_embeddings, z_name_expanded, chunk_scores
                torch.cuda.empty_cache()
            
            # 获取推荐的RGB值（需要更多候选以便去重）
            recommended_indices = global_top_k_indices.cpu().numpy()
            
            # 提取前3维作为RGB值
            candidate_rgbs = []
            for idx in recommended_indices:
                rgb = unique_colors_array[idx]
                if len(rgb) >= 3:
                    candidate_rgbs.append(rgb[:3])
                else:
                    candidate_rgbs.append(list(rgb) + [0.0] * (3 - len(rgb)))
            
            candidate_rgbs = np.array(candidate_rgbs)
            
            # 对候选颜色进行LAB距离去重，确保推荐的颜色之间LAB距离大于2.3
            unique_rgb_values = []
            unique_scores = []
            
            # 遍历所有候选颜色进行去重
            for i, (rgb, score) in enumerate(zip(candidate_rgbs, global_top_k_values.cpu().numpy())):
                if len(unique_rgb_values) == 0:
                    # 第一个颜色直接添加
                    unique_rgb_values.append(rgb)
                    unique_scores.append(score)
                else:
                    # 计算与所有已去重颜色的LAB距离
                    current_lab = rgb_to_lab(rgb)
                    existing_labs = np.array([rgb_to_lab(existing_rgb) for existing_rgb in unique_rgb_values])
                    distances = np.sqrt(np.sum((current_lab - existing_labs) ** 2, axis=1))
                    
                    # 如果最小距离大于阈值2.3，则添加
                    if np.min(distances) >= 2.3:
                        unique_rgb_values.append(rgb)
                        unique_scores.append(score)
                    
                    # 如果已经找到足够的去重颜色，停止
                    if len(unique_rgb_values) >= top_k:
                        break
            
            rgb_values = np.array(unique_rgb_values)
            raw_scores = np.array(unique_scores)
            # 分数校准，增强区分度
            # scores = calibrate_scores(raw_scores, gamma=2.5, alpha=0.88)
            scores = raw_scores
        
        return rgb_values, scores
        
    except Exception as e:
        print(f"Error recommending RGB colors: {e}")
        import traceback
        traceback.print_exc()
        return [], []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_color_name', methods=['POST'])
def predict_color_name():
    try:
        data = request.get_json()
        r = float(data['r']) / 255.0
        g = float(data['g']) / 255.0
        b = float(data['b']) / 255.0
        
        rgb_input = [r, g, b]
        names, scores = recommend_color_names(rgb_input, top_k=10)
        
        results = []
        for name, score in zip(names, scores):
            results.append({
                'name': name,
                'score': float(score)
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/generate_quick_color', methods=['POST'])
def generate_quick_color_api():
    """快速生成一个颜色并立即返回"""
    try:
        data = request.get_json()
        color_name = data['color_name'].lower().strip()
        
        generated_rgb = generate_quick_color(color_name)
        
        if generated_rgb is not None:
            # 转换回0-255范围
            r = int(generated_rgb[0] * 255)
            g = int(generated_rgb[1] * 255)
            b = int(generated_rgb[2] * 255)
            
            result = {
                'rgb': [r, g, b],
                'hex': f"#{r:02x}{g:02x}{b:02x}",
                'score': 1.0,
                'is_generated': True
            }
            
            return jsonify({
                'success': True,
                'result': result
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Unable to generate quick color'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/recommend_colors', methods=['POST'])
def recommend_colors():
    """推荐更多颜色（在快速生成后调用）"""
    try:
        data = request.get_json()
        color_name = data['color_name'].lower().strip()
        
        rgb_values, scores = recommend_rgb_colors(color_name, top_k=10)
        
        results = []
        for rgb, score in zip(rgb_values, scores):
            # 转换回0-255范围
            r = int(rgb[0] * 255)
            g = int(rgb[1] * 255)
            b = int(rgb[2] * 255)
            
            results.append({
                'rgb': [r, g, b],
                'hex': f"#{r:02x}{g:02x}{b:02x}",
                'score': float(score),
                'is_generated': False
            })
        
        return jsonify({
            'success': True,
            'results': results
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

if __name__ == '__main__':
        import argparse
        parser = argparse.ArgumentParser(description='Color Name Web App')
        parser.add_argument('--device', choices=['cpu', 'gpu'], default='gpu', help='选择运行设备，默认gpu')
        parser.add_argument('--port', type=int, default=5000, help='服务端口，默认5000')
        args = parser.parse_args()

        # 解析设备
        if args.device == 'cpu':
            selected_device = torch.device('cpu')
        else:
            selected_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print("Starting to load model...")
        if load_model(device=selected_device):
            print("Model loaded successfully, starting Flask app...")
            
            # 根据环境变量决定是否开启调试模式
            debug_mode = os.getenv('FLASK_DEBUG', '0') == '1'
            print(f"Debug mode: {'ON' if debug_mode else 'OFF'}")
            
            app.run(debug=debug_mode, host='0.0.0.0', port=args.port)
        else:
            print("Model loading failed, please check model files")