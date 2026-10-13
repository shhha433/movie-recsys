"""
统计「召回结果里热门物品的占比」
====================================
用途：检查模型的召回结果是不是过度偏向热门物品。

为什么重要：如果模型只是"把最热门的电影推给所有人"，Recall 也能上去——
因为测试集里的正样本本身就偏热门。这种"提升"是假的，上线会伤害长尾和多样性。

跑法：
    /opt/miniconda3/bin/python outputs/week3/hot_ratio.py

注意：这个文件只 import numpy + faiss，不要加 torch（OpenMP 冲突会崩）。
"""

import os

import faiss
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
WEEK3 = _HERE
MOVIES_DAT = os.path.join(_HERE, "data", "ml-1m", "movies.dat")
TRAIN_CSV = os.path.join(_HERE, "data", "processed", "train.csv")

TOP_N = 100      # "热门"的定义：训练集里被点击次数最多的前 100 部电影
K = 50           # 召回 top-K
EXPS = ["emb_baseline_full", "emb_cosineB_full", "emb_cosineB2_full"]


def main():
    # ------------------------------------------------------------
    # 第 1 步：读 movies.dat，拿到【全部电影 ID】，并从小到大排序
    # ------------------------------------------------------------
    # 为什么要排序？因为训练时的 item2idx 就是按这个顺序编号的：
    #     item2idx = {movie_id: i+1 for i, movie_id in enumerate(sorted(全部电影ID))}
    # 所以 item_vectors.npy 的第 i 行 ↔ sorted_ids[i]
    # 不排序的话，下面"向量行号 ↔ 电影 ID"的对应关系就全错了。
    ids = sorted(
        int(line.split("::")[0])
        for line in open(MOVIES_DAT, encoding="latin-1")
    )
    print(f"全库电影数：{len(ids)}")

    # ------------------------------------------------------------
    # 第 2 步：统计每部电影的热度 = 它在训练集里作为【正样本】出现的次数
    # ------------------------------------------------------------
    # 只读两列，省内存（train.csv 有 227 万行）
    train = pd.read_csv(
        TRAIN_CSV,
        usecols=["target_item_id", "label"],
    )
    # label==1 的是正样本；value_counts() 数出每部电影出现了多少次
    pop = train[train.label == 1]["target_item_id"].value_counts()
    # 热度最高的 TOP_N 部，装进一个 set（后面判断"在不在里面"用，O(1) 查找）
    hot_ids = set(pop.head(TOP_N).index)
    print(f"top{TOP_N} 热门片占全库比例：{TOP_N / len(ids):.2%}")
    print()

    # ------------------------------------------------------------
    # 第 3 步：逐个实验算「召回的 top-K 里，热门片占多少」
    # ------------------------------------------------------------
    header = f"{'实验':<22}{'Recall@50':>10}{'top-50 里热门片占比':>20}"
    print(header)
    print("-" * len(header))

    for name in EXPS:
        d = os.path.join(WEEK3, name)
        if not os.path.exists(os.path.join(d, "item_vectors.npy")):
            print(f"{name:<22}{'（缺文件，跳过）':>32}")
            continue

        # 物品向量：[物品数, d]，第 i 行是 sorted_ids[i] 的向量
        item = np.load(f"{d}/item_vectors.npy").astype("float32")
        # 用户向量：每个"测试集正样本"的用户向量
        user = np.load(f"{d}/valid_user_vectors.npy").astype("float32")
        # 目标：每个正样本对应的目标物品（item2idx 编号，1 开始）
        tgt = np.load(f"{d}/valid_targets.npy")
        keep = tgt > 0                 # 映射失败的（编号 0）丢掉
        user, tgt = user[keep], tgt[keep]

        # 建 Faiss 索引：IndexFlatIP = 内积检索（暴力但精确，3883 个物品足够快）
        index = faiss.IndexFlatIP(item.shape[1])
        # ascontiguousarray：Faiss 要求内存连续，numpy 切片后常常不连续，不加会崩
        index.add(np.ascontiguousarray(item))
        # 检索：返回每个用户最相似的 K 个物品的【位置】和分数
        _, pos = index.search(np.ascontiguousarray(user), K)

        # ---- Recall@50：目标物品在不在召回的 K 个里 ----
        # Faiss 返回的是位置（0 开始），而 tgt 存的是 item2idx（1 开始），所以减 1
        hit = np.mean([
            (int(t) - 1) in set(row.tolist())
            for row, t in zip(pos, tgt)
        ])

        # ---- 热门占比 ----
        # 对每一行（每个用户）的 K 个位置，数其中有多少个是热门片
        #     ids[p] 把"向量行号"翻译回"电影 ID"
        #     ∈ hot_ids 判断是不是 top100
        #     除以 K 得到这一行的比例
        hot_per_row = [
            sum(1 for p in row if ids[p] in hot_ids) / K
            for row in pos
        ]
        print(f"{name:<22}{hit:>10.4f}{np.mean(hot_per_row):>20.2%}")

    print()
    print("怎么读这张表：")
    print("  · 全库热门片只占 2.58%，如果召回的 top-50 里热门占了 50%，说明严重偏热门")
    print("  · Recall 高 + 热门占比也高 → 指标提升里有一部分是'靠推热门'，不一定是好事")


if __name__ == "__main__":
    main()
