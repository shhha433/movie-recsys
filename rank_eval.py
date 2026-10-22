"""
召回 → 排序 端到端评估

把召回层和排序层真正串起来，回答两个问题：
    1. 直接用召回层的分数排序，效果如何？
    2. 让排序模型（DIN）在召回的候选里重新排序，能不能变好？

流程：
    1. 读召回层导出的向量（EMB_DIR）
    2. 对验证集里每个用户，从全库物品里取召回分数最高的 50 个 —— 这就是「召回」
    3. 两组排序：
           A. 直接用召回分数排序（也就是召回的原始顺序）
           B. 用 DIN 给这 50 个候选重新打分、重新排序
    4. 对比 HR@10 / NDCG@10 / MRR

指标怎么读：
    Recall@50   天花板——目标物品有没有进候选集。没进来的，排序层再准也救不回来
    HR@10       端到端——目标最终有没有排进前 10
    NDCG@10     不只要求进前 10，还要求排得更靠前
    MRR         目标排名倒数的平均值

注意：这个脚本只 import numpy + torch，**不用 faiss**。
不是偷懒——macOS 上 faiss-cpu 和 PyTorch 各带一份 OpenMP，同进程会直接崩
（退出码 134/139，且没有 Python 报错，见 README 第九节）。
而 3883 个物品的暴力检索只要几秒，根本不需要 ANN；
只有在工业级的全库规模下，Faiss 才是必需的。

跑法（必须先跑 din.py，它会存下 model_din.pt）：
    /opt/miniconda3/bin/python rank_eval.py
"""

import os
import sys
import time

import numpy as np
import torch

import two_tower as tt
from din import DIN

_HERE = os.path.dirname(os.path.abspath(__file__))

# 必须和 two_tower.py 的 EMB_OUT_DIR 指向同一个目录：
# 召回候选、训练难负样本、端到端评估要来自同一个召回模型，否则又会对不齐。
EMB_DIR = os.path.join(_HERE, "emb_baseline_full")
CKPT = os.path.join(_HERE, "model_din.pt")           # din.py 训练后保存的排序模型
TOPK = 50        # 召回候选数
RANK_K = 10      # 排序后取前几个
CHUNK = 8192     # 检索分块大小
BATCH = 512      # 排序打分时的批大小（每个用户带 50 个候选，别设太大）


def recall_candidates(user_vec, item_vec, topk=TOPK):
    """向量暴力检索 top-k。

    item_vec 的第 i 行对应 item2idx == i+1，所以返回值已经 +1 换回 item2idx 空间。
    返回 (候选索引 [n, topk], 召回分数 [n, topk])，都已按分数从高到低排好——
    这个顺序就是「直接用召回结果排序」的结果。
    """
    n = len(user_vec)
    cand = np.empty((n, topk), dtype=np.int64)
    scores = np.empty((n, topk), dtype=np.float32)
    for start in range(0, n, CHUNK):
        end = min(start + CHUNK, n)
        sim = user_vec[start:end] @ item_vec.T                     # [b, n_items]
        part = np.argpartition(-sim, topk, axis=1)[:, :topk]
        part_score = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-part_score, axis=1)
        cand[start:end] = np.take_along_axis(part, order, axis=1) + 1
        scores[start:end] = np.take_along_axis(part_score, order, axis=1)
    return cand, scores


def rank_metrics(cand, targets, k=RANK_K):
    """cand 的每一行必须已经按分数从高到低排好。targets 是 item2idx 空间的索引。

    目标没被召回的样本记 0 分（端到端口径），这样指标才反映真实线上效果。
    """
    n = len(targets)
    hit = np.zeros(n)
    ndcg = np.zeros(n)
    rr = np.zeros(n)
    for i in range(n):
        pos = np.flatnonzero(cand[i] == targets[i])
        if pos.size == 0:
            continue
        rank = int(pos[0]) + 1
        rr[i] = 1.0 / rank
        if rank <= k:
            hit[i] = 1.0
            ndcg[i] = 1.0 / np.log2(rank + 1)      # 只有一个相关物品，所以 IDCG = 1
    return hit.mean(), ndcg.mean(), rr.mean()


@torch.no_grad()
def score_with_din(model, hist, cand, device, batch=BATCH):
    """用排序模型给每个用户的 topk 个候选逐个打分，返回 [n, topk]。

    关键点：DIN 的用户向量依赖候选，所以每个 (用户, 候选) 对都要重算一次，
    不能像双塔那样把用户向量算一次就复用。
    """
    model.eval()
    n, topk = cand.shape
    length = hist.shape[1]
    out = np.empty((n, topk), dtype=np.float32)
    hist_t = torch.tensor(hist, dtype=torch.long)

    for start in range(0, n, batch):
        end = min(start + batch, n)
        seq = hist_t[start:end].to(device)                          # [b, L]
        items = torch.tensor(cand[start:end], dtype=torch.long,
                             device=device)                         # [b, topk]
        b = seq.shape[0]
        seq_rep = seq.unsqueeze(1).expand(b, topk, length).reshape(b * topk, length)
        user_vec, item_vec = model(seq_rep, items.reshape(-1))
        out[start:end] = (user_vec * item_vec).sum(dim=1).reshape(b, topk).cpu().numpy()
    return out


def main():
    if not os.path.exists(CKPT):
        print(f"找不到排序模型：{CKPT}")
        print("先跑 din.py 训练它。")
        sys.exit(1)

    device = tt.get_device()
    print(f"设备      ：{device}")
    print(f"召回向量  ：{os.path.basename(EMB_DIR)}")
    print(f"排序模型  ：{os.path.basename(CKPT)}")
    print()

    # ---- 1. 读召回层导出的向量 ----
    item_vec = np.load(os.path.join(EMB_DIR, "item_vectors.npy")).astype("float32")
    user_vec = np.load(os.path.join(EMB_DIR, "valid_user_vectors.npy")).astype("float32")
    targets = np.load(os.path.join(EMB_DIR, "valid_targets.npy"))
    keep = targets > 0                      # 只保留目标物品在词表里的样本
    user_vec, targets = user_vec[keep], targets[keep]
    n_items_all = len(item_vec) + 1         # 含 padding 位的物品总数

    item2idx, n_items = tt.build_vocab()
    if n_items != n_items_all:
        print(f"物品数对不上：向量文件 {n_items_all}，词表 {n_items}。")
        print("向量可能是在另一份数据上导出的，先重跑 two_tower.py。")
        sys.exit(1)

    # 验证集正样本的历史序列（顺序必须和 valid_targets.npy 一致）
    valid_df = tt.load_split("valid")
    pos = valid_df[valid_df["label"] == 1].reset_index(drop=True)
    hist = tt.build_history_matrix(pos, item2idx)[keep]

    print(f"验证集用户数：{len(targets)}    物品数：{n_items_all}")
    print()

    # ---- 2. 召回 ----
    t0 = time.time()
    cand, _ = recall_candidates(user_vec, item_vec)
    in_cand = (cand == targets[:, None]).any(axis=1)
    ceiling = in_cand.mean()
    print(f"召回 top-{TOPK} 完成（{time.time() - t0:.1f}s）")
    print(f"  召回天花板 Recall@{TOPK}  = {ceiling:.4f}    ← 目标物品进候选集的比例")
    print()

    # ---- 3A. 直接用召回分数排序 ----
    hr_a, ndcg_a, mrr_a = rank_metrics(cand, targets)

    # ---- 3B. 用 DIN 重新排序 ----
    print("加载排序模型并对候选打分 ...")
    model = DIN(n_items).to(device)
    model.load_state_dict(torch.load(CKPT, map_location=device))

    t0 = time.time()
    score_b = score_with_din(model, hist, cand, device)
    print(f"  打分完成（{time.time() - t0:.1f}s）")
    order = np.argsort(-score_b, axis=1)
    cand_b = np.take_along_axis(cand, order, axis=1)
    hr_b, ndcg_b, mrr_b = rank_metrics(cand_b, targets)
    print()

    # ---- 4. 对比 ----
    print("=" * 72)
    print(f"{'排序方式':<26}{'HR@10':>10}{'NDCG@10':>12}{'MRR':>10}")
    print("-" * 72)
    print(f"{'A. 直接用召回分数':<26}{hr_a:>10.4f}{ndcg_a:>12.4f}{mrr_a:>10.4f}")
    print(f"{'B. DIN 重新排序':<26}{hr_b:>10.4f}{ndcg_b:>12.4f}{mrr_b:>10.4f}")
    print("=" * 72)
    print(f"{'差值（B - A）':<26}{hr_b - hr_a:>+10.4f}"
          f"{ndcg_b - ndcg_a:>+12.4f}{mrr_b - mrr_a:>+10.4f}")
    print()
    print("口径：分母是全部验证集用户，目标没被召回的记 0 分。")
    print()

    if in_cand.any():
        print(f"只看被召回的 {int(in_cand.sum())} 个样本（条件口径）：")
        hr_a_c, ndcg_a_c, mrr_a_c = rank_metrics(cand[in_cand], targets[in_cand])
        hr_b_c, ndcg_b_c, mrr_b_c = rank_metrics(cand_b[in_cand], targets[in_cand])
        print(f"  A. 直接用召回分数   HR@10 = {hr_a_c:.4f}   NDCG@10 = {ndcg_a_c:.4f}"
              f"   MRR = {mrr_a_c:.4f}")
        print(f"  B. DIN 重新排序     HR@10 = {hr_b_c:.4f}   NDCG@10 = {ndcg_b_c:.4f}"
              f"   MRR = {mrr_b_c:.4f}")


if __name__ == "__main__":
    main()
