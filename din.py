
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---- 复用已经写好的工具函数 ----
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                        # repo/ 布局：和 two_tower.py 同目录
from two_tower import (  # noqa: E402
    MAX_LEN, EMB_DIM, build_vocab, build_history_matrix,
    TwoTowerDataset, evaluate_auc, set_seed, get_device, load_split,
)

# ============================================================
# 配置
# ============================================================
BATCH_SIZE = 2048
EPOCHS = 10      # 之前 3 轮远没收敛（valid AUC 每轮还在涨），改成 10 轮
LR = 1e-3
SEED = 42
MAX_ROWS = 0      # 0 = 全量；调试先用 200000

# 排序层的训练数据：第 3.5 步生成难负样本后优先用它，否则退回随机负样本
_TRAIN_HARD = os.path.join(_HERE, "data", "processed", "train_hard.csv")
TRAIN_SPLIT = "train_hard" if os.path.exists(_TRAIN_HARD) else "train"


# ============================================================
# DIN 模型
# ============================================================
class DIN(nn.Module):
    def __init__(self, n_items, emb_dim=EMB_DIM, pad_idx=0):
        super().__init__()
        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=pad_idx)
        # 注意力打分网络：吃「候选 / 历史 / 两者之差」三段拼接，输出一个分数
        self.att_mlp = nn.Sequential(
            nn.Linear(emb_dim * 3, 32), nn.ReLU(), nn.Linear(32, 1)
        )
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )

    def attention(self, user_seq, item_idx):
        user_emb=self.item_emb(user_seq)#user_emb:[B,L,d]
        item_emb=self.item_emb(item_idx).unsqueeze(1).expand(-1,user_seq.size(1),-1)#[B,1,d]->[B,L,d]
        concat=torch.cat([user_emb,item_emb,user_emb-item_emb],dim=-1)#[B,L,3d]
        scores = self.att_mlp(concat).squeeze(-1).float()   # [b, l]
        mask=(user_seq==0)
        scores = scores.masked_fill(mask, float('-inf'))
        att=torch.softmax(scores,dim=1)
        att=torch.nan_to_num(att,nan=0.0)
        return att



    def forward(self, user_seq, item_idx):
        att=self.attention(user_seq,item_idx).unsqueeze(dim=-1)
        user_emb=self.item_emb(user_seq)
        user_vec=self.user_mlp((user_emb*att).sum(dim=1))
        item_vec=self.item_mlp(self.item_emb(item_idx))
        return user_vec,item_vec


# ============================================================
# 自动自检（不用改）
# ============================================================
def check_din(model, seq, item):
    """检查 attention 是否满足两个硬条件。"""
    problems = []
    try:
        att = model.attention(seq, item)
    except NotImplementedError:
        return ["attention() 还是 NotImplementedError"]
    except Exception as e:
        return [f"attention() 报错：{type(e).__name__}: {e}"]

    if tuple(att.shape) != tuple(seq.shape):
        problems.append(f"attention 形状应为 {tuple(seq.shape)}，实际 {tuple(att.shape)}")
        return problems

    pad_pos = (seq == 0)
    if pad_pos.any() and att[pad_pos].abs().max().item() > 1e-6:
        worst = att[pad_pos].abs().max().item()
        problems.append(f"padding 位置的注意力权重不是 0（最大 {worst:.4f}）"
                        "——softmax 之前忘了 masked_fill(-inf)？")

    row_sum = att.sum(dim=1)
    if (row_sum - 1).abs().max().item() > 1e-4:
        problems.append(f"每行权重之和应该是 1，实际在 "
                        f"{row_sum.min().item():.4f} ~ {row_sum.max().item():.4f} 之间")

    if torch.isnan(att).any():
        problems.append("attention 里有 nan")

    return problems


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for seq, item, label in loader:
        seq, item, label = seq.to(device), item.to(device), label.to(device)
        optimizer.zero_grad()
        user_vec, item_vec = model(seq, item)
        score = (user_vec * item_vec).sum(dim=1, keepdim=True)
        loss = loss_fn(score, label)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)


def main():
    set_seed()
    device = get_device()
    print(f"设备：{device}")

    train_df = load_split(TRAIN_SPLIT, MAX_ROWS)
    valid_df = load_split("valid")
    test_df = load_split("test")
    print(f"训练数据：{TRAIN_SPLIT}.csv"
          f"{'（难负样本）' if TRAIN_SPLIT == 'train_hard' else '（随机负样本）'}")
    print(f"train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)}")

    item2idx, n_items = build_vocab()
    train_hist = build_history_matrix(train_df, item2idx)
    train_set = TwoTowerDataset(
        train_hist,
        train_df["target_item_id"].map(lambda x: item2idx.get(int(x), 0)).to_numpy(),
        train_df["label"].to_numpy(dtype=np.float32),
    )
    valid_set = TwoTowerDataset(
        build_history_matrix(valid_df, item2idx),
        valid_df["target_item_id"].map(lambda x: item2idx.get(int(x), 0)).to_numpy(),
        valid_df["label"].to_numpy(dtype=np.float32),
    )
    test_set = TwoTowerDataset(
        build_history_matrix(test_df, item2idx),
        test_df["target_item_id"].map(lambda x: item2idx.get(int(x), 0)).to_numpy(),
        test_df["label"].to_numpy(dtype=np.float32),
    )

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=4096)
    test_loader = DataLoader(test_set, batch_size=4096)

    model = DIN(n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")

    # ---- 先跑自检，再训练 ----
    seq, item, _ = next(iter(DataLoader(train_set, batch_size=64)))
    problems = check_din(model, seq.to(device), item.to(device))
    if problems:
        print()
        print("自检没通过：")
        for p in problems:
            print("  [问题] " + p)
        return
    print("自检通过 ✅（padding 权重为 0、每行权重和为 1）")
    print()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        auc = evaluate_auc(model, valid_loader, device)
        print(f"epoch {epoch}  train_loss={loss:.4f}  valid_AUC={auc:.4f}")

    print()
    test_auc = evaluate_auc(model, test_loader, device)
    print(f"test AUC = {test_auc:.4f}")
    print()
    print("对照：双塔（均值池化）的 valid AUC = 0.8678、test AUC = 0.8511")

    # 把训练好的排序模型存下来，交给 rank_eval.py 做「召回 → 排序」端到端评估
    ckpt_path = os.path.join(_HERE, "model_din.pt")
    torch.save(model.state_dict(), ckpt_path)
    print()
    print(f"模型已保存：{ckpt_path}")
    print("接着跑这一条，把召回和排序串起来评估：")
    print(f"    /opt/miniconda3/bin/python {os.path.join(_HERE, 'rank_eval.py')}")


if __name__ == "__main__":
    main()
