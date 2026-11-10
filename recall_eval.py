import os
import sys
import argparse

import numpy as np
import faiss

_HERE = os.path.dirname(os.path.abspath(__file__))
WEEK3_DIR = _HERE

RECALL_K = 50         # 想算 Recall@10 就改成 10
N_BOOT = 1000         # bootstrap 重采样次数
SEED = 42
BASELINE_NAME = "emb_baseline_full"   # 本次以全量 baseline 为基准

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="+", default=None)
    parser.add_argument("--baseline", default="emb_baseline_full")
    return parser.parse_args()

def find_experiments(selected=None):
    """扫描所有 emb*/ 目录，返回排好序的目录名列表。"""
    exps = []
    if selected:
        missing = [
            name for name in selected
            if not os.path.exists(
                os.path.join(WEEK3_DIR, name, "item_vectors.npy")
            )
        ]
        if missing:
            raise FileNotFoundError(
                f"找不到这些实验的 item_vectors.npy：{missing}"
            )
        return list(selected)
    for name in sorted(os.listdir(WEEK3_DIR)):
        d = os.path.join(WEEK3_DIR, name)
        if name.startswith("emb") and os.path.isdir(d) \
                and os.path.exists(os.path.join(d, "item_vectors.npy")):
            exps.append(name)
    return exps


def hit_vector(emb_dir, k=RECALL_K, cache=True):
    """返回 (命中向量, 物品数, 目标数组)。命中向量里 1 表示召回的 top-k 里有它。"""
    n_items_all = len(np.load(os.path.join(emb_dir, "item_vectors.npy")))
    cache_path = os.path.join(emb_dir, f"hits@{k}.npy")
    # 缓存只有在【比向量文件新】的时候才可用，否则说明向量被重训过了，必须重算
    vec_files = [os.path.join(emb_dir, f) for f in
                 ("item_vectors.npy", "valid_user_vectors.npy", "valid_targets.npy")]
    cache_fresh = (os.path.exists(cache_path)
                   and os.path.getmtime(cache_path) > max(os.path.getmtime(f) for f in vec_files))
    if cache and cache_fresh:
        hits = np.load(cache_path)
        tgt = np.load(os.path.join(emb_dir, "valid_targets.npy"))
        return hits, n_items_all, tgt[tgt > 0]

    item = np.load(os.path.join(emb_dir, "item_vectors.npy")).astype("float32")
    user = np.load(os.path.join(emb_dir, "valid_user_vectors.npy")).astype("float32")
    tgt = np.load(os.path.join(emb_dir, "valid_targets.npy"))
    keep = tgt > 0
    user, tgt = user[keep], tgt[keep]

    index = faiss.IndexFlatIP(item.shape[1])
    index.add(np.ascontiguousarray(item))
    _, pos = index.search(np.ascontiguousarray(user), k)

    # Faiss 返回的是索引里的位置：位置 0 对应 item2idx == 1，所以目标要减 1
    hits = np.zeros(len(tgt), dtype=np.float64)
    for i, (row, t) in enumerate(zip(pos, tgt)):
        if (int(t) - 1) in set(row.tolist()):
            hits[i] = 1.0
    if cache:
        np.save(cache_path, hits)
    return hits, n_items_all, tgt


def bootstrap_ci(hits, n_boot=N_BOOT, alpha=0.05, rng=None):
    rng = rng or np.random.default_rng(SEED)
    n = len(hits)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        boots[b] = hits[idx].mean()
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return hits.mean(), lo, hi


def paired_bootstrap(hits_a, hits_b, n_boot=N_BOOT, seed=SEED):
    """对同一批重采样索引算两个实验的差值，返回 (点估计, 下界, 上界, p 值)。"""
    if len(hits_a) != len(hits_b):
        return None      # 评估集不一样，不能做配对比较
    rng = np.random.default_rng(seed)
    n = len(hits_a)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = hits_b[idx].mean() - hits_a[idx].mean()
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    p = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return hits_b.mean() - hits_a.mean(), lo, hi, p


def main():
    args = parse_args()
    exps = find_experiments(args.experiments)
    if not exps:
        print(f"{WEEK3_DIR} 下没找到任何 emb*/ 目录。")
        print("先跑 two_tower.py（它会把向量导出到 EMB_OUT_DIR）。")
        sys.exit(1)

    if args.baseline not in exps:
        print(f"基准 {args.baseline!r} 不在实验列表中：{exps}")
        sys.exit(1)

    base = args.baseline
    ordered = [base] + [e for e in exps if e != base]

    print(f"发现 {len(exps)} 个实验：")
    for e in exps:
        print(f"  {e}")
    print(f"\n基准实验（用来做对比）= {base}\n")

    print("=" * 74)
    print(f"{'实验':<20}{'Recall@' + str(RECALL_K):>12}{'95% 置信区间':>26}{'命中 / 总数':>18}")
    print("=" * 74)

    targets_all = {}
    hits_all, n_items = {}, None
    for e in ordered:
        d = os.path.join(WEEK3_DIR, e)
        hits, n_items_all, tgt = hit_vector(d)
        targets_all[e] = tgt
        n_items = n_items_all
        point, lo, hi = bootstrap_ci(hits)
        hits_all[e] = hits
        print(f"{e:<20}{point:>12.4f}{f'[{lo:.4f}, {hi:.4f}]':>26}"
              f"{f'{int(hits.sum())} / {len(hits)}':>18}")

    print("=" * 74)
    print(f"随机基线参考值 = {RECALL_K}/{n_items} = {RECALL_K / n_items:.4f}")
    print()

    if len(ordered) == 1:
        print("只有一个实验。想比较的话，在 two_tower.py 里换个 EMB_OUT_DIR 再跑一次。")
        return

    print("配对 bootstrap（每个实验 vs 基准）：")
    print("-" * 74)
    print(f"{'实验':<20}{'差异':>10}{'差异 95% 置信区间':>26}{'p 值':>10}   结论")
    print("-" * 74)
    for e in ordered:
        if not np.array_equal(targets_all[base], targets_all[e]):
            print(f"{e:<20}{'—':>10}{'目标顺序不同，跳过':>26}{'—':>10}   不能比较")
            continue
        if e == base:
            continue
        res = paired_bootstrap(hits_all[base], hits_all[e])
        if res is None:
            print(f"{e:<20}{'—':>10}{'评估集不同，跳过':>26}{'—':>10}   不能比较")
            continue
        diff, lo, hi, p = res
        if lo > 0 or hi < 0:
            verdict = "显著"
        elif abs(diff) < 1e-9:
            verdict = "完全一致（预期如此）"
        else:
            verdict = "不显著，不能下结论"
        print(f"{e:<20}{diff:>+10.4f}{f'[{lo:+.4f}, {hi:+.4f}]':>26}{p:>10.4f}   {verdict}")

    print()
    print("判读方法：差异的置信区间【跨过 0】= 这个差异不能排除是噪声，别急着下结论。")


if __name__ == "__main__":
    main()
