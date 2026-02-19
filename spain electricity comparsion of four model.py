import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler
from torch import autocast 

# ==========================================
# 0. 核心组件 (模型层)
# ==========================================
class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super(Transpose, self).__init__()
        self.dim0, self.dim1 = dim0, dim1
    def forward(self, x): return x.transpose(self.dim0, self.dim1)

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5):
        super().__init__()
        self.affine_weight = nn.Parameter(torch.ones(num_features))
        self.affine_bias = nn.Parameter(torch.zeros(num_features))
        self.eps = eps
    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            return (x - self.mean) / self.stdev * self.affine_weight + self.affine_bias
        return (x - self.affine_bias) / (self.affine_weight + self.eps) * self.stdev + self.mean

# ==========================================
# 1. Swin 1D 核心逻辑
# ==========================================
class SwinWindowAttention1D(nn.Module):
    def __init__(self, dim, window_size, shift_size, num_heads=4):
        super().__init__()
        self.dim, self.window_size, self.shift_size, self.num_heads = dim, window_size, shift_size, num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.rel_pos_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.rel_pos_bias_table, std=.02)

    def forward(self, x, mask=None):
        B_N, num_win, ws, D = x.shape
        qkv = self.qkv(x).reshape(B_N, num_win, ws, 3, self.num_heads, D // self.num_heads).permute(3, 0, 1, 4, 2, 5)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        coords = torch.arange(ws); rel_index = (coords[None, :] - coords[:, None] + ws - 1).to(x.device)
        rel_pos_bias = self.rel_pos_bias_table[rel_index.view(-1)].view(ws, ws, -1).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)
        attn = attn + rel_pos_bias
        if mask is not None: attn = attn + mask.unsqueeze(1).unsqueeze(0)
        attn = F.softmax(attn, dim=-1)
        return self.proj((attn @ v).transpose(2, 3).reshape(B_N, num_win, ws, D))

class SwinBlock1D(nn.Module):
    def __init__(self, dim, seq_len, window_size=8, shift_size=4):
        super().__init__()
        self.dim, self.window_size, self.shift_size = dim, window_size, shift_size
        self.norm1 = nn.LayerNorm(dim); self.attn = SwinWindowAttention1D(dim, window_size, 0)
        self.norm2 = nn.LayerNorm(dim); self.shift_attn = SwinWindowAttention1D(dim, window_size, shift_size)
        if self.shift_size > 0:
            img_mask = torch.zeros((1, seq_len, 1)); s_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            for i, s in enumerate(s_slices): img_mask[:, s, :] = i
            m_win = img_mask.view(1, seq_len // window_size, window_size, 1).reshape(-1, window_size)
            attn_mask = m_win.unsqueeze(1) - m_win.unsqueeze(2)
            self.register_buffer("attn_mask", attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0))
        else: self.attn_mask = None

    def forward(self, x):
        B_N, L, D = x.shape
        h = x; x = self.norm1(x).view(B_N, L // self.window_size, self.window_size, D)
        x = self.attn(x).view(B_N, L, D) + h
        h = x; x = self.norm2(x)
        if self.shift_size > 0: x = torch.roll(x, shifts=-self.shift_size, dims=1)
        x = x.view(B_N, L // self.window_size, self.window_size, D)
        x = self.shift_attn(x, mask=self.attn_mask).view(B_N, L, D)
        if self.shift_size > 0: x = torch.roll(x, shifts=self.shift_size, dims=1)
        return x + h

# ==========================================
# 2. 五专家门控系统
# ==========================================
class IntraColumnPentaExperts(nn.Module):
    def __init__(self, seq_len=96, d_model=256, path_drop=0.35):
        super().__init__()
        self.path_drop = path_drop
        self.exp_global = nn.Linear(seq_len, d_model)
        self.exp_local = nn.Sequential(
            nn.Conv1d(1, 64, 3, padding=1), nn.GELU(), 
            nn.Conv1d(64, 1, 3, padding=1), nn.Flatten(), 
            nn.Linear(seq_len, d_model), nn.Dropout(0.5)
        )
        self.exp_diff = nn.Sequential(
            nn.Linear(seq_len - 1, d_model), nn.LayerNorm(d_model), nn.Dropout(0.45)
        )
        self.exp_swin = nn.Sequential(
            nn.Linear(1, d_model), SwinBlock1D(d_model, seq_len), 
            Transpose(1, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.45)
        )
        self.exp_sliding = nn.Sequential(
            nn.Conv1d(1, d_model, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(d_model), nn.GELU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Dropout(0.45)
        )
        self.gate = nn.Sequential(nn.Linear(d_model * 5, 128), nn.GELU(), nn.Linear(128, 5))
        self.temp = 2.0 

    def forward(self, x_strip):
        f1 = self.exp_global(x_strip)
        f2 = self.exp_local(x_strip.unsqueeze(1))
        f3 = self.exp_diff(x_strip[:, 1:] - x_strip[:, :-1])
        f4 = self.exp_swin(x_strip.unsqueeze(-1))
        f5 = self.exp_sliding(x_strip.unsqueeze(1))
        experts = [f1, f2, f3, f4, f5]
        if self.training:
            for i in range(len(experts)):
                if torch.rand(1) < self.path_drop: experts[i] = torch.zeros_like(experts[i])
        logits = self.gate(torch.cat([f1, f2, f3, f4, f5], dim=-1))
        w = F.softmax(logits / self.temp, dim=-1)
        out = sum(w[:, i:i+1] * experts[i] for i in range(5))
        return out, w

# ==========================================
# 3. 终极模型架构
# ==========================================
class XManifoldUltra(nn.Module):
    def __init__(self, seq_len=96, num_vars=9, d_model=256, noise_std=0.01):
        super().__init__()
        self.noise_std = noise_std
        self.revin = RevIN(num_vars)
        self.experts = IntraColumnPentaExperts(seq_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, 8, d_model*4, 0.2, batch_first=True, norm_first=True)
        self.game_room = nn.TransformerEncoder(encoder_layer, 3)
        self.head = nn.Sequential(nn.Linear(num_vars * d_model, 512), nn.GELU(), nn.Dropout(0.45), nn.Linear(512, 1))

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin(x, 'norm')
        tokens, gate_w = self.experts(x.permute(0, 2, 1).reshape(B * N, L))
        tokens = tokens.view(B, N, -1)
        if self.training: tokens = tokens + torch.randn_like(tokens) * self.noise_std
        refined = self.game_room(tokens)
        return self.head(refined.reshape(B, -1)), gate_w

# ==========================================
# 4. 数据加载 (零泄露严苛版)
# ==========================================
class SpainDataset(Dataset):
    def __init__(self, csv_path, flag='train', seq_len=96, split_ratio=0.8):
        full_df = pd.read_csv(csv_path)
        
        # 1. 特征工程与标签预生成
        target_col = 'price actual'
        actual_cols = [
            'price actual', 'total load actual', 'generation wind onshore', 
            'generation solar', 'generation fossil gas', 'generation nuclear', 
            'generation fossil hard coal'
        ]
        
        # 强制特征滞后 1 位 (Zero-Leakage 核心)
        df_features = full_df[actual_cols].shift(1)
        
        # 时间特征 (周期性编码)
        if 'time' in full_df.columns:
            t = pd.to_datetime(full_df['time'], utc=True)
            hour = t.dt.hour
        else:
            hour = pd.Series(np.arange(len(full_df)) % 24)
        df_features['h_sin'] = np.sin(2 * np.pi * hour / 24)
        df_features['h_cos'] = np.cos(2 * np.pi * hour / 24)
        
        # 价格标签生成
        raw_price = pd.to_numeric(full_df[target_col], errors='coerce').interpolate().ffill().bfill().values
        labels = (raw_price[1:] > raw_price[:-1]).astype(float)
        
        # 2. 严格按比例拆分
        split_idx = int(len(full_df) * split_ratio)
        if flag == 'train':
            feat_subset = df_features.iloc[1:split_idx].copy()
            label_subset = labels[0 : split_idx-1]
        else:
            feat_subset = df_features.iloc[split_idx - seq_len : ].copy()
            label_subset = labels[split_idx - seq_len - 1 : ]

        # 3. 在子集内部进行插值
        data_list = []
        for c in feat_subset.columns:
            col = pd.to_numeric(feat_subset[c], errors='coerce').interpolate().ffill().bfill()
            data_list.append(col.values)
        
        self.data = np.stack(data_list, axis=1).astype(np.float32)
        self.labels = label_subset.astype(np.float32)
        self.seq_len = seq_len
        
    def __len__(self): return len(self.data) - self.seq_len
    def __getitem__(self, i): 
        return torch.tensor(self.data[i : i+self.seq_len]), \
               torch.tensor(self.labels[i+self.seq_len-1]).unsqueeze(-1)

# ==========================================
# 5. 训练系统
# ==========================================
def smoothed_bce_loss(logits, labels, smoothing=0.40):
    with torch.no_grad():
        labels = labels * (1 - smoothing) + 0.5 * smoothing
    return F.binary_cross_entropy_with_logits(logits, labels)

def main():
    if "LOCAL_RANK" not in os.environ: return
    rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl")
    
    train_ds = SpainDataset('energy_dataset.csv', 'train')
    val_ds = SpainDataset('energy_dataset.csv', 'val')
    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler = DistributedSampler(val_ds, shuffle=False)
    
    train_loader = DataLoader(train_ds, 128, sampler=train_sampler, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, 128, sampler=val_sampler, num_workers=4, pin_memory=True)

    model = XManifoldUltra(96, 9, 256).to(rank)
    model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)
    scaler = GradScaler()

    if rank == 0: print(f"🚀 ZERO-LEAKAGE Edition | Smoothing: 0.40")

    best_acc = 0.0
    for epoch in range(20):
        model.train()
        train_sampler.set_epoch(epoch)
        pbar = tqdm(train_loader, disable=(rank != 0), desc=f"Epoch {epoch+1}")
        for bx, by in pbar:
            bx, by = bx.to(rank), by.to(rank)
            optimizer.zero_grad()
            with autocast(device_type='cuda'):
                logits, _ = model(bx)
                loss = smoothed_bce_loss(logits, by)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        scheduler.step()
        model.eval()
        v_acc, v_cnt = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(rank), by.to(rank)
                logits, _ = model(bx)
                v_acc += ((torch.sigmoid(logits) > 0.5) == by).sum().item()
                v_cnt += by.size(0)

        stats = torch.tensor([v_acc, v_cnt]).to(rank)
        dist.all_reduce(stats)
        if rank == 0:
            current_acc = stats[0].item() / stats[1].item()
            best_acc = max(best_acc, current_acc)
            print(f"📊 Val Acc: {current_acc:.2%} | Best: {best_acc:.2%}")

    dist.destroy_process_group()

if __name__ == "__main__": main()

(base) root@ubuntu22:~# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test3.py
[2026-02-13 02:31:51,129] torch.distributed.run: [WARNING] 
[2026-02-13 02:31:51,129] torch.distributed.run: [WARNING] *****************************************
[2026-02-13 02:31:51,129] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-02-13 02:31:51,129] torch.distributed.run: [WARNING] *****************************************
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:286: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
/root/anaconda3/lib/python3.11/site-packages/torch/nn/modules/transformer.py:286: UserWarning: enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.norm_first was True
  warnings.warn(f"enable_nested_tensor is True, but self.use_nested_tensor is False because {why_not_sparsity_fast_path}")
🚀 ZERO-LEAKAGE Edition | Smoothing: 0.40
Epoch 1: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 28.40it/s]
📊 Val Acc: 77.50% | Best: 77.50%
Epoch 2: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.83it/s]
📊 Val Acc: 80.64% | Best: 80.64%
Epoch 3: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.48it/s]
📊 Val Acc: 80.65% | Best: 80.65%
Epoch 4: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.34it/s]
📊 Val Acc: 80.78% | Best: 80.78%
Epoch 5: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.49it/s]
📊 Val Acc: 80.90% | Best: 80.90%
Epoch 6: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.24it/s]
📊 Val Acc: 81.11% | Best: 81.11%
Epoch 7: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.38it/s]
📊 Val Acc: 81.32% | Best: 81.32%
Epoch 8: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.31it/s]
📊 Val Acc: 81.55% | Best: 81.55%
Epoch 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.19it/s]
📊 Val Acc: 81.19% | Best: 81.55%
Epoch 10: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.86it/s]
📊 Val Acc: 81.57% | Best: 81.57%
Epoch 11: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.55it/s]
📊 Val Acc: 81.42% | Best: 81.57%
Epoch 12: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.27it/s]
📊 Val Acc: 81.79% | Best: 81.79%
Epoch 13: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.76it/s]
📊 Val Acc: 81.89% | Best: 81.89%
Epoch 14: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.16it/s]
📊 Val Acc: 81.48% | Best: 81.89%
Epoch 15: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.76it/s]
📊 Val Acc: 82.09% | Best: 82.09%
Epoch 16: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.38it/s]
📊 Val Acc: 81.82% | Best: 82.09%
Epoch 17: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 34.28it/s]
📊 Val Acc: 82.02% | Best: 82.09%
Epoch 18: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.61it/s]
📊 Val Acc: 81.95% | Best: 82.09%
Epoch 19: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.80it/s]
📊 Val Acc: 81.88% | Best: 82.09%
Epoch 20: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 110/110 [00:03<00:00, 33.41it/s]
📊 Val Acc: 81.84% | Best: 82.09%













import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. 官方组件：RevIN (可逆实例归一化)
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
        return x

# ==========================================
# 2. iTransformer 官方核心架构
# ==========================================
class iTransformer_Official(nn.Module):
    def __init__(self, n_vars, seq_len, d_model=128, n_heads=8, e_layers=3, d_ff=256, dropout=0.1):
        super().__init__()
        self.revin = RevIN(n_vars)
        # 维度反转：将 L 长度的时间维度嵌入到 D 维空间
        self.enc_embedding = nn.Linear(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, 
            dropout=dropout, batch_first=True, activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_vars * d_model, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        # x: [B, L, N]
        x = self.revin(x, 'norm')
        B, L, N = x.shape
        
        # Inversion: [B, L, N] -> [B, N, L]
        x = x.permute(0, 2, 1)
        
        # Variate Embedding
        enc_out = self.enc_embedding(x) # [B, N, D]
        
        # Variate Attention
        enc_out = self.encoder(enc_out) # [B, N, D]
        
        return self.head(enc_out)

# ==========================================
# 3. 严苛数据引擎 (零泄露)
# ==========================================
class SpainDataset(Dataset):
    def __init__(self, csv_path, flag='train', seq_len=96, scaler=None):
        df = pd.read_csv(csv_path)
        # 选定的物理特征
        actual_cols = [
            'price actual', 'total load actual', 'generation wind onshore', 
            'generation solar', 'generation fossil gas', 'generation nuclear'
        ]
        
        # 1. 特征滞后 1 位，确保预测 T 时只有 T-1 之前的数据
        df_feat = df[actual_cols].shift(1).ffill().bfill()
        # 2. 生成涨跌标签
        prices = df['price actual'].ffill().bfill().values
        labels = (prices[1:] > prices[:-1]).astype(float)
        
        n = len(df_feat) - 1
        split_idx = int(n * 0.8)
        
        if flag == 'train':
            feat_raw = df_feat.iloc[:split_idx].values
            self.labels = labels[:split_idx]
            self.scaler = StandardScaler().fit(feat_raw)
        else:
            feat_raw = df_feat.iloc[split_idx:].values
            self.labels = labels[split_idx:]
            self.scaler = scaler # 验证集必须用训练集的 scaler
            
        self.feat = self.scaler.transform(feat_raw)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.feat) - self.seq_len

    def __getitem__(self, i):
        return torch.tensor(self.feat[i : i+self.seq_len], dtype=torch.float32), \
               torch.tensor([self.labels[i+self.seq_len-1]], dtype=torch.float32)

# ==========================================
# 4. DDP 训练系统
# ==========================================
def main():
    # 初始化 DDP
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 参数设置
    CSV_PATH = 'energy_dataset.csv'
    SEQ_LEN = 96
    SMOOTHING = 0.1 # 👈 标签平滑

    # 数据准备
    train_ds = SpainDataset(CSV_PATH, 'train', SEQ_LEN)
    val_ds = SpainDataset(CSV_PATH, 'val', SEQ_LEN, scaler=train_ds.scaler)
    
    train_sampler = DistributedSampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=128, sampler=train_sampler)
    val_loader = DataLoader(val_ds, batch_size=128)

    # 模型实例化
    model = iTransformer_Official(n_vars=6, seq_len=SEQ_LEN).to(device)
    model = DDP(model, device_ids=[local_rank])
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    if rank == 0: print(f"🚀 官方 iTransformer 启动 | 模式: 零泄露 + 标签平滑")

    for epoch in range(20):
        model.train()
        train_sampler.set_epoch(epoch)
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            
            # 标签平滑处理
            with torch.no_grad():
                by_smooth = by * (1 - SMOOTHING) + 0.5 * SMOOTHING
            
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by_smooth)
            loss.backward()
            optimizer.step()

        # 验证命中率
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = (logits > 0).float()
                correct += (preds == by).sum().item()
                total += by.size(0)

        # 汇总所有显卡的统计结果
        stats = torch.tensor([correct, total]).to(device)
        dist.all_reduce(stats)
        
        if rank == 0:
            acc = stats[0].item() / stats[1].item()
            print(f"Epoch [{epoch+1:02d}] | 命中准确率 (Val Acc): {acc:.2%}")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()

(base) root@ubuntu22:~# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test3.py
[2026-02-13 02:57:46,816] torch.distributed.run: [WARNING] 
[2026-02-13 02:57:46,816] torch.distributed.run: [WARNING] *****************************************
[2026-02-13 02:57:46,816] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-02-13 02:57:46,816] torch.distributed.run: [WARNING] *****************************************
🚀 官方 iTransformer 启动 | 模式: 零泄露 + 标签平滑
Epoch [01] | 命中准确率 (Val Acc): 77.33%
Epoch [02] | 命中准确率 (Val Acc): 79.94%
Epoch [03] | 命中准确率 (Val Acc): 80.70%
Epoch [04] | 命中准确率 (Val Acc): 80.83%
Epoch [05] | 命中准确率 (Val Acc): 80.82%
Epoch [06] | 命中准确率 (Val Acc): 80.93%
Epoch [07] | 命中准确率 (Val Acc): 80.83%
Epoch [08] | 命中准确率 (Val Acc): 80.96%
Epoch [09] | 命中准确率 (Val Acc): 80.73%
Epoch [10] | 命中准确率 (Val Acc): 80.64%
Epoch [11] | 命中准确率 (Val Acc): 81.08%
Epoch [12] | 命中准确率 (Val Acc): 80.85%
Epoch [13] | 命中准确率 (Val Acc): 80.83%
Epoch [14] | 命中准确率 (Val Acc): 81.17%
Epoch [15] | 命中准确率 (Val Acc): 80.44%
Epoch [16] | 命中准确率 (Val Acc): 80.47%
Epoch [17] | 命中准确率 (Val Acc): 80.56%
Epoch [18] | 命中准确率 (Val Acc): 80.31%
Epoch [19] | 命中准确率 (Val Acc): 80.40%
Epoch [20] | 命中准确率 (Val Acc): 80.51%




import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.preprocessing import StandardScaler

# ==========================================
# 0. 官方核心组件：RevIN
# ==========================================
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.affine_weight = nn.Parameter(torch.ones(num_features))
            self.affine_bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x, mode: str):
        if mode == 'norm':
            self.mean = x.mean(1, keepdim=True).detach()
            self.stdev = torch.sqrt(x.var(1, keepdim=True, unbiased=False) + self.eps).detach()
            x = (x - self.mean) / self.stdev
            if self.affine: x = x * self.affine_weight + self.affine_bias
        elif mode == 'denorm':
            if self.affine: x = (x - self.affine_bias) / self.affine_weight
            x = x * self.stdev + self.mean
        return x

# ==========================================
# 1. 四大官方模型定义
# ==========================================

# --- ① PatchTST Official ---
class PatchTST_Official(nn.Module):
    def __init__(self, n_vars, seq_len, patch_len=16, stride=8, d_model=128, n_heads=8, e_layers=3):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.n_patches = int((seq_len - patch_len) / stride + 2)
        self.revin = RevIN(n_vars)
        self.padding = nn.ReplicationPad1d((0, stride))
        self.W_P = nn.Linear(patch_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model*4, dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(n_vars * self.n_patches * d_model, 1))

    def forward(self, x):
        x = self.revin(x, 'norm')
        B, L, N = x.shape
        x = x.permute(0, 2, 1)
        x = self.padding(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(B * N, -1, self.patch_len)
        x = self.encoder(self.W_P(x))
        x = x.reshape(B, N, -1, x.shape[-1])
        return self.head(x)

# --- ② TimesNet Official ---
class TimesNet_Official(nn.Module):
    def __init__(self, n_vars, seq_len, d_model=64, top_k=5):
        super().__init__()
        self.seq_len, self.k = seq_len, top_k
        self.revin = RevIN(n_vars)
        self.conv = nn.ModuleList([nn.Sequential(
            nn.Conv2d(n_vars, d_model, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(d_model, n_vars, kernel_size=3, padding=1)
        ) for _ in range(top_k)])
        self.projection = nn.Linear(n_vars * seq_len, 1)

    def forward(self, x):
        x = self.revin(x, 'norm')
        B, L, N = x.shape
        xf = torch.fft.rfft(x, dim=1)
        amplitudes = torch.abs(xf).mean(0).mean(-1)
        amplitudes[0] = 0
        _, top_list = torch.topk(amplitudes, self.k)
        periods = L // top_list
        res = []
        for i in range(self.k):
            period = periods[i]
            length = ((L // period) + 1) * period
            out = torch.cat([x, torch.zeros([B, length - L, N]).to(x.device)], dim=1)
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2)
            out = self.conv[i](out)
            res.append(out.reshape(B, N, length).permute(0, 2, 1)[:, :L, :])
        return self.projection(torch.stack(res, dim=-1).sum(-1).reshape(B, -1))

# --- ③ TSMixer Official ---
class TSMixer_Official(nn.Module):
    def __init__(self, n_vars, seq_len, dropout=0.1):
        super().__init__()
        self.revin = RevIN(n_vars)
        self.temp_mix = nn.Sequential(nn.LayerNorm([n_vars, seq_len]), nn.Linear(seq_len, seq_len), nn.GELU(), nn.Dropout(dropout))
        self.feat_mix = nn.Sequential(nn.LayerNorm([seq_len, n_vars]), nn.Linear(n_vars, n_vars), nn.GELU(), nn.Dropout(dropout))
        self.head = nn.Linear(seq_len * n_vars, 1)

    def forward(self, x):
        x = self.revin(x, 'norm')
        x = x + self.temp_mix(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + self.feat_mix(x)
        return self.head(x.reshape(x.shape[0], -1))

# --- ④ TimeMixer Official ---
class TimeMixer_Official(nn.Module):
    def __init__(self, n_vars, seq_len, d_model=128):
        super().__init__()
        self.revin = RevIN(n_vars)
        self.down = nn.AvgPool1d(kernel_size=3, padding=1, stride=2)
        self.proj = nn.Linear(seq_len + seq_len // 2, 256)
        self.head = nn.Linear(256 * n_vars, 1)

    def forward(self, x):
        B, L, N = x.shape
        x = self.revin(x, 'norm')
        x_low = self.down(x.permute(0, 2, 1)).permute(0, 2, 1)
        combined = torch.cat([x, x_low], dim=1)
        out = F.gelu(self.proj(combined.permute(0, 2, 1)))
        return self.head(out.reshape(B, -1))

# ==========================================
# 2. 严苛数据加载
# ==========================================
class SpainDataset(Dataset):
    def __init__(self, csv_path, flag='train', seq_len=96, scaler=None):
        df = pd.read_csv(csv_path)
        cols = ['price actual', 'total load actual', 'generation wind onshore', 'generation solar', 'generation fossil gas', 'generation nuclear']
        df_feat = df[cols].shift(1).ffill().bfill()
        labels = (df['price actual'].ffill().bfill().values[1:] > df['price actual'].ffill().bfill().values[:-1]).astype(float)
        split = int((len(df_feat)-1) * 0.8)
        if flag == 'train':
            feat_raw = df_feat.iloc[:split].values
            self.labels = labels[:split]
            self.scaler = StandardScaler().fit(feat_raw)
        else:
            self.labels = labels[split:]
            self.scaler = scaler
            feat_raw = df_feat.iloc[split:].values
        self.feat = self.scaler.transform(feat_raw)
        self.seq_len = seq_len

    def __len__(self): return len(self.feat) - self.seq_len
    def __getitem__(self, i):
        return torch.tensor(self.feat[i:i+self.seq_len], dtype=torch.float32), torch.tensor([self.labels[i+self.seq_len-1]], dtype=torch.float32)

# ==========================================
# 3. 统一 DDP 训练引擎
# ==========================================
def main():
    dist.init_process_group("nccl")
    rank, local_rank = int(os.environ["RANK"]), int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    # 可在此切换模型进行对比: "PatchTST", "TimesNet", "TSMixer", "TimeMixer"
    MODEL_NAME = "TSMixer" 
    SEQ_LEN, SMOOTHING = 96, 0.1

    train_ds = SpainDataset('energy_dataset.csv', 'train', SEQ_LEN)
    val_ds = SpainDataset('energy_dataset.csv', 'val', SEQ_LEN, scaler=train_ds.scaler)
    train_loader = DataLoader(train_ds, batch_size=128, sampler=DistributedSampler(train_ds))
    val_loader = DataLoader(val_ds, batch_size=128)

    if MODEL_NAME == "PatchTST": model = PatchTST_Official(6, SEQ_LEN)
    elif MODEL_NAME == "TimesNet": model = TimesNet_Official(6, SEQ_LEN)
    elif MODEL_NAME == "TSMixer": model = TSMixer_Official(6, SEQ_LEN)
    else: model = TimeMixer_Official(6, SEQ_LEN)

    model = DDP(model.to(device), device_ids=[local_rank])
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    if rank == 0: print(f"🚀 正在运行官方版: {MODEL_NAME} | 标签平滑: {SMOOTHING}")

    for epoch in range(20):
        model.train()
        train_loader.sampler.set_epoch(epoch)
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            by_s = by * (1 - SMOOTHING) + 0.5 * SMOOTHING
            optimizer.zero_grad(); loss = criterion(model(bx), by_s); loss.backward(); optimizer.step()

        model.eval(); correct, total = 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                correct += ((model(bx) > 0).float() == by).sum().item(); total += by.size(0)
        
        stats = torch.tensor([correct, total]).to(device)
        dist.all_reduce(stats)
        if rank == 0: print(f"Epoch {epoch+1:02d} | Val Acc: {stats[0].item()/stats[1].item():.2%}")

    dist.destroy_process_group()

if __name__ == "__main__": main()








(base) root@ubuntu22:~# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test2.py
[2026-02-13 03:00:20,944] torch.distributed.run: [WARNING] 
[2026-02-13 03:00:20,944] torch.distributed.run: [WARNING] *****************************************
[2026-02-13 03:00:20,944] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-02-13 03:00:20,944] torch.distributed.run: [WARNING] *****************************************
🚀 正在运行官方版: PatchTST | 标签平滑: 0.1
Epoch 01 | Val Acc: 79.71%
Epoch 02 | Val Acc: 80.20%
Epoch 03 | Val Acc: 81.02%
Epoch 04 | Val Acc: 80.98%
Epoch 05 | Val Acc: 80.46%
Epoch 06 | Val Acc: 80.95%
Epoch 07 | Val Acc: 80.50%
Epoch 08 | Val Acc: 80.72%
Epoch 09 | Val Acc: 80.98%
Epoch 10 | Val Acc: 80.95%
Epoch 11 | Val Acc: 81.18%
Epoch 12 | Val Acc: 80.96%
Epoch 13 | Val Acc: 81.24%
Epoch 14 | Val Acc: 81.19%
Epoch 15 | Val Acc: 80.85%
Epoch 16 | Val Acc: 81.14%
Epoch 17 | Val Acc: 80.75%
Epoch 18 | Val Acc: 80.95%
Epoch 19 | Val Acc: 80.69%
Epoch 20 | Val Acc: 81.28%


(base) root@ubuntu22:~# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test2.py 
[2026-02-13 03:01:17,359] torch.distributed.run: [WARNING] 
[2026-02-13 03:01:17,359] torch.distributed.run: [WARNING] *****************************************
[2026-02-13 03:01:17,359] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-02-13 03:01:17,359] torch.distributed.run: [WARNING] *****************************************
🚀 正在运行官方版: TimesNet | 标签平滑: 0.1
Epoch 01 | Val Acc: 75.34%
Epoch 02 | Val Acc: 76.50%
Epoch 03 | Val Acc: 77.05%
Epoch 04 | Val Acc: 77.77%
Epoch 05 | Val Acc: 78.25%
Epoch 06 | Val Acc: 78.64%
Epoch 07 | Val Acc: 78.78%
Epoch 08 | Val Acc: 79.26%
Epoch 09 | Val Acc: 79.21%
Epoch 10 | Val Acc: 79.56%
Epoch 11 | Val Acc: 79.63%
Epoch 12 | Val Acc: 79.63%
Epoch 13 | Val Acc: 79.75%
Epoch 14 | Val Acc: 79.91%
Epoch 15 | Val Acc: 79.85%
Epoch 16 | Val Acc: 79.91%
Epoch 17 | Val Acc: 79.86%
Epoch 18 | Val Acc: 79.84%
Epoch 19 | Val Acc: 79.91%
Epoch 20 | Val Acc: 80.07%


(base) root@ubuntu22:~# CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 test2.py 
[2026-02-13 03:02:24,664] torch.distributed.run: [WARNING] 
[2026-02-13 03:02:24,664] torch.distributed.run: [WARNING] *****************************************
[2026-02-13 03:02:24,664] torch.distributed.run: [WARNING] Setting OMP_NUM_THREADS environment variable for each process to be 1 in default, to avoid your system being overloaded, please further tune the variable for optimal performance in your application as needed. 
[2026-02-13 03:02:24,664] torch.distributed.run: [WARNING] *****************************************
🚀 正在运行官方版: TSMixer | 标签平滑: 0.1
Epoch 01 | Val Acc: 73.75%
Epoch 02 | Val Acc: 75.09%
Epoch 03 | Val Acc: 76.08%
Epoch 04 | Val Acc: 76.61%
Epoch 05 | Val Acc: 77.09%
Epoch 06 | Val Acc: 77.46%
Epoch 07 | Val Acc: 77.94%
Epoch 08 | Val Acc: 78.27%
Epoch 09 | Val Acc: 78.55%
Epoch 10 | Val Acc: 78.78%
Epoch 11 | Val Acc: 78.90%
Epoch 12 | Val Acc: 79.17%
Epoch 13 | Val Acc: 79.52%
Epoch 14 | Val Acc: 79.91%
Epoch 15 | Val Acc: 79.66%
Epoch 16 | Val Acc: 80.05%
Epoch 17 | Val Acc: 80.33%
Epoch 18 | Val Acc: 80.25%
Epoch 19 | Val Acc: 80.34%
Epoch 20 | Val Acc: 80.34%
(base) root@ubuntu22:~# 