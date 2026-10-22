"""
从召回候选里挖难负样本，生成排序层的训练数据
召回层和排序层如果各训各的，会出现「训练/推理候选分布不一致」：
    训练时负样本是全库随机抽的 —— 随便一部电影，和正样本差得远，闭着眼能分开
    推理时候选是召回 top-50 —— 全是"用户可能喜欢"的，要在鸡蛋里挑骨头
模型没学过后者，端到端评估自然打不过"直接用召回分数排序"。

这一步把训练数据换成「召回 top-K 里挑出来的负样本」，让训练和推理面对同一种难度。

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


def pick_random(history, target, all_ids, k=NEG, rng=None):
    """目标没被召回时的兜底：从全库随机抽负样本。

    如果正样本根本不在 top-K 里，那 top-K 全是
    "召回模型认为用户会喜欢的物品"。把它们当负样本，等于教模型
    "召回候选 = 负"，信号会整个反过来（实测 AUC 掉到 0.53）。
    """
    rng = rng or np.random
    seen = set(int(x) for x in history if x != 0)
    picked = []
    while len(picked) < k:
        c = int(rng.choice(all_ids))
        if c == target or c in seen or c in picked:
            continue
        picked.append(c)
    return picked


def main():
    if not os.path.exists(CKPT):
        print(f"找不到召回模型：{CKPT}")
        print("先跑 two_tower.py（它现在会保存 model_two_tower.pt）。")
        sys.exit(1)

    device = tt.get_device()
    print(f"设备：{device}")

    item2idx, n_items = tt.build_vocab()
    # 反向表：连续索引 → 原始电影 id。写 csv 必须用原始 id，
    # 否则 din.py 会把索引当成 id 再映射一次，训练数据整体错位。
    idx2item = {v: k for k, v in item2idx.items()}
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
    raw_targets = pos["target_item_id"].astype(int).to_numpy()   # 原始电影 id，写进 csv 用这个
    histories = pos["history"].fillna("").astype(str).to_numpy()
    user_ids = pos["user_id"].to_numpy()
    all_raw = np.array(list(item2idx.keys()))   # 随机兜底用
    n_hard = n_random = 0

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
                writer.writerow([uid, hist_str, int(raw_targets[row]), 1])
                # 只有目标真的被召回了，top-K 才是"和正样本同难度的候选"，才能当难负样本；
                # 否则 top-K 全是召回模型喜欢的东西，拿来当负样本会把信号教反。
                if target in set(int(c) for c in cands[i]):
                    negs = [idx2item[x] for x in pick_negatives(cands[i], target, hist[i])]
                    n_hard += 1
                else:
                    negs = pick_random(hist[i], target, all_raw)
                    n_random += 1
                for neg in negs:
                    writer.writerow([uid, hist_str, int(neg), 0])
                n_written += 1 + NEG

            done = end
            print(f"  {done}/{len(pos)}  {time.time() - t0:.0f}s", flush=True)

    print()
    print(f"已写出：{OUT_PATH}")
    print(f"行数：{n_written}（正样本 {n_written // (1 + NEG)}）")
    print(f"负样本来源：难负样本 {n_hard} 组 / 随机负样本 {n_random} 组"
          f"（目标被召回的才用难负样本）")
    print()
    print("接着跑 din.py —— 它检测到 train_hard.csv 就会自动用它训练。")


if __name__ == "__main__":
    main()
