"""
第 3.5 步：从召回候选里挖难负样本，生成排序层的训练数据

为什么需要这一步
----------------
召回层和排序层如果各训各的，会出现「训练/推理候选分布不一致」：
    训练时负样本是全库随机抽的 —— 随便一部电影，和正样本差得远，闭着眼能分开
    推理时候选是召回 top-50 —— 全是"用户可能喜欢"的，要在鸡蛋里挑骨头
模型没学过后者，端到端评估自然打不过"直接用召回分数排序"。

这一步把训练数据换成「召回 top-K 里挑出来的负样本」，让训练和推理面对同一种难度。

产物
----
    data/processed/train_hard.csv    列和 train.csv 完全一样
                                     （user_id, history, target_item_id, label）
                                     每个正样本配 4 个难负样本，总数和 train.csv 相同

依赖
----
    必须先跑 two_tower.py，它会保存 model_two_tower.pt。

跑法
----
    /opt/miniconda3/bin/python build_hard_negatives.py
    全量约 3-5 分钟；调参时把 MAX_ROWS 设成 200000 先试。
"""

import csv
import os
import sys
import time

import numpy as np
import torch

import two_tower as tt
from two_tower import TwoTower

_HERE = os.path.dirname(os.path.abspath(__file__))

CKPT = os.path.join(_HERE, "model_two_tower.pt")
OUT_PATH = os.path.join(_HERE, "data", "processed", "train_hard.csv")

TOPK = 50         # 召回候选数：从这 K 个里挑负样本
NEG = 4           # 每个正样本配几个负样本（和 train.csv 的 1:4 保持一致）
MAX_ROWS = 0      # 0 = 全量；调试时设成 200000
CHUNK = 4096      # 检索分块大小


@torch.no_grad()
def get_item_vectors(model, n_items, device):
    """把 1..n_items-1 号物品的向量算出来，返回 [n_items-1, d]。"""
    out = []
    for chunk in torch.split(torch.arange(1, n_items, device=device), CHUNK):
        dummy_seq = torch.zeros(chunk.shape[0], tt.MAX_LEN, dtype=torch.long, device=device)
        _, item_vec = model(dummy_seq, chunk)
        out.append(item_vec.cpu().numpy())
    return np.concatenate(out).astype("float32")


@torch.no_grad()
def get_user_vectors(model, hist, device, batch=4096):
    """把历史序列矩阵算成用户向量，返回 [n, d]。"""
    out = []
    for start in range(0, len(hist), batch):
        seq = torch.tensor(hist[start:start + batch], dtype=torch.long, device=device)
        dummy_item = torch.ones(seq.shape[0], dtype=torch.long, device=device)
        user_vec, _ = model(seq, dummy_item)
        out.append(user_vec.cpu().numpy())
    return np.concatenate(out).astype("float32")


def pick_negatives(cands, target, history, k=NEG):
    """从召回候选里挑 k 个负样本：排除目标物品和用户已经看过的。"""
    seen = set(int(x) for x in history if x != 0)
    picked = []
    for c in cands:
        c = int(c)
        if c == target or c in seen:
            continue
        picked.append(c)
        if len(picked) == k:
            break
    if len(picked) < k:                        # 理论上不会发生（K=50 远大于 3+4）
        raise RuntimeError(f"候选不够：只挑到 {len(picked)} 个")
    return picked


def main():
    if not os.path.exists(CKPT):
        print(f"找不到召回模型：{CKPT}")
        print("先跑 two_tower.py（它现在会保存 model_two_tower.pt）。")
        sys.exit(1)

    device = tt.get_device()
    print(f"设备：{device}")

    item2idx, n_items = tt.build_vocab()
    train_df = tt.load_split("train", MAX_ROWS)
    pos = train_df[train_df["label"] == 1].reset_index(drop=True)
    print(f"训练集正样本数：{len(pos)}（每个配 {NEG} 个难负样本）")

    model = TwoTower(n_items).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))
    model.eval()

    t0 = time.time()
    item_vec = get_item_vectors(model, n_items, device)
    hist = tt.build_history_matrix(pos, item2idx)
    user_vec = get_user_vectors(model, hist, device)
    print(f"向量算完（{time.time() - t0:.1f}s）：物品 {item_vec.shape}，用户 {user_vec.shape}")

    targets = pos["target_item_id"].map(lambda x: item2idx.get(int(x), 0)).to_numpy()
    histories = pos["history"].fillna("").astype(str).to_numpy()
    user_ids = pos["user_id"].to_numpy()

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    t0 = time.time()
    n_written = 0

    with open(OUT_PATH, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["user_id", "history", "target_item_id", "label"])

        for start in range(0, len(pos), CHUNK):
            end = min(start + CHUNK, len(pos))
            sim = user_vec[start:end] @ item_vec.T              # [b, n_items]
            part = np.argpartition(-sim, TOPK, axis=1)[:, :TOPK]
            part_score = np.take_along_axis(sim, part, axis=1)
            order = np.argsort(-part_score, axis=1)
            cands = np.take_along_axis(part, order, axis=1) + 1  # 换回 item2idx

            for i in range(end - start):
                row = start + i
                uid = int(user_ids[row])
                hist_str = histories[row]
                target = int(targets[row])
                if target <= 0:
                    continue
                writer.writerow([uid, hist_str, target, 1])
                for neg in pick_negatives(cands[i], target, hist[i]):
                    writer.writerow([uid, hist_str, neg, 0])
                n_written += 1 + NEG

            done = end
            print(f"  {done}/{len(pos)}  {time.time() - t0:.0f}s", flush=True)

    print()
    print(f"已写出：{OUT_PATH}")
    print(f"行数：{n_written}（正样本 {n_written // (1 + NEG)}）")
    print()
    print("接着跑 din.py —— 它检测到 train_hard.csv 就会自动用它训练。")


if __name__ == "__main__":
    main()
