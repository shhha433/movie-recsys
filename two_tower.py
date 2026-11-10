
import os
import sys
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# ============================================================
# 配置
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))                                            # wo
DATA_DIR = os.path.join(_HERE, "data","processed")
MOVIES_PATH = os.path.join(_HERE, "data", "ml-1m", "movies.dat")

MAX_LEN = 20          # history 最多保留最近多少个物品（必须 ≥ MAX_HISTORY，否则会被截断）
EMB_DIM = 32          # 每层 embedding 的维度
BATCH_SIZE = 2048
EPOCHS = 10      # 之前 3 轮远没收敛（valid AUC 每轮还在涨），改成 10 轮
LR = 1e-3
# 默认输出目录；实际运行时可用 --out-dir 覆盖
EMB_OUT_DIR = os.path.join(_HERE, "emb_mask_genre")
SEED = 42
MAX_ROWS = 0      # 0 = 用全部；调试时设成 200000

# ---- 默认实验配置，实际运行时由命令行参数覆盖 ----
POOLING = "mask"     # "mean"：全部历史求平均 | "last"：取最近一个 | "mask"：写法不同但等价于 mean
USE_COSINE = False   # True：训练时把用户/物品向量归一化，打分变成余弦
USE_GENRES = True    # 物品向量 = ID embedding + genre projection


def set_seed(seed=None):
    seed = SEED if seed is None else seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pooling",
        choices=("mean", "last", "mask"),
        default=POOLING,
    )
    parser.add_argument(
        "--genres",
        action=argparse.BooleanOptionalAction,
        default=USE_GENRES,
    )
    parser.add_argument(
        "--cosine",
        action=argparse.BooleanOptionalAction,
        default=USE_COSINE,
    )
    parser.add_argument("--out-dir", default=os.path.basename(EMB_OUT_DIR))
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 1. 读数据与建词表
# ============================================================
def load_split(name, max_rows=None):
    """读某一份数据。max_rows 只用来缩小【训练集】做快速调试——
    验证集和测试集永远用全量，否则不同实验之间的指标没法比较。"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"找不到 {path}")
        print("先跑 prepare_data.py 生成三个 csv。")
        sys.exit(1)
    df = pd.read_csv(path)
    if max_rows:
        df = df.sample(n=min(max_rows, len(df)), random_state=SEED).reset_index(drop=True)
    return df


def build_vocab():
    """item_id → 连续索引。0 号位留给 padding，真实物品从 1 开始。
    直接用 movies.dat 里的全部电影建表，保证 history 里出现的任何物品都有 embedding。
    """
    item_ids = []
    with open(MOVIES_PATH, encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("::")
            if len(parts) >= 3:
                item_ids.append(int(parts[0]))
    item2idx = {iid: i + 1 for i, iid in enumerate(sorted(item_ids))}
    return item2idx, len(item_ids) + 1

def build_genre_matrix(item2idx):
    """按 item2idx 的顺序，构造 [n_items, 18] 的多热矩阵；第 0 行全 0。"""
    genres = {}
    with open(MOVIES_PATH, encoding="latin-1") as fh:
        for line in fh:
            parts = line.strip().split("::")
            if len(parts) < 3:
                continue
            genres[int(parts[0])] = parts[2].split("|")

    vocab = sorted({g for gs in genres.values() for g in gs})
    g2i = {g: i for i, g in enumerate(vocab)}
    mat = np.zeros((len(item2idx) + 1, len(vocab)), dtype=np.float32)
    for iid, idx in item2idx.items():
        for g in genres.get(iid, []):
            mat[idx, g2i[g]] = 1.0
    return mat

# ============================================================
# 2. 把 history 字符串转成定长整数矩阵
# ============================================================
def build_history_matrix(df, item2idx, max_len=MAX_LEN):
    n=len(df)
    matrix = np.zeros((n,max_len),dtype=np.int64)
    for i, hist in enumerate(df["history"].fillna("").astype(str)):
        if not hist.strip():
            continue  # 整行都是 padding
        # 1) 拆分 + 取最近 max_len 个
        tokens = np.array(hist.split(),dtype=int)
        tokens = tokens[-max_len:]
        # 2) 映射成索引
        idxs = [item2idx[t] for t in tokens if t in item2idx]
        # 3) 只保留最近 max_len 个（防止映射后变少，但一般不会）
        idxs = idxs[-max_len:]
        # 4) 左补 0
        if idxs:
            matrix[i, max_len - len(idxs):] = idxs
    return matrix


class HistoryMatrixError(Exception):
    pass


def validate_history_matrix(df, mat, item2idx, max_len=MAX_LEN):
    """验证 history 的 ID 映射、顺序和 padding 是否符合约定。"""
    problems = []
    if not isinstance(mat, np.ndarray):
        return [f"返回值应该是 numpy 数组，实际是 {type(mat).__name__}"]
    if mat.shape != (len(df), max_len):
        return [f"形状应该是 {(len(df), max_len)}，实际是 {mat.shape}"]
    if not np.issubdtype(mat.dtype, np.integer):
        problems.append(f"dtype 应该是整数类型，实际是 {mat.dtype}")

    sample = df.head(2000)
    for row_i, hist in enumerate(sample["history"].tolist()):
        ids = str(hist).split() if hist else []
        ids = ids[-max_len:]                      # 只留最近 max_len 个
        want = [item2idx.get(int(x), 0) for x in ids]
        got = [int(v) for v in mat[row_i]]
        got_nonzero = [v for v in got if v != 0]
        want_nonzero = [v for v in want if v != 0]
        if got_nonzero != want_nonzero:
            problems.append(
                f"第 {row_i} 行对不上：history={hist!r}，期望非零部分 {want_nonzero}，实际 {got_nonzero}"
            )
            break
        if got.count(0) != max_len - len(want_nonzero):
            problems.append(f"第 {row_i} 行 padding 数量不对：实际 {got}")
            break

    if problems:
        for p in problems:
            print("  [问题] " + p)
    return problems


# ============================================================
# 3. Dataset
# ============================================================
class TwoTowerDataset(Dataset):
    def __init__(self, history_mat, item_idx, labels):
        self.history_mat = history_mat
        self.item_idx = item_idx
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return (
            torch.tensor(self.history_mat[i], dtype=torch.long),
            torch.tensor(self.item_idx[i], dtype=torch.long),
            torch.tensor(self.labels[i], dtype=torch.float32).unsqueeze(-1),
        )


def build_dataset(df, item2idx, item_mat=None):
    """把 DataFrame 变成 Dataset 需要的三个数组。"""
    if item_mat is None:
        hist_mat = build_history_matrix(df, item2idx)
    else:
        hist_mat = item_mat
    ids = df["target_item_id"].map(lambda x: item2idx.get(int(x), 0)).to_numpy()
    labels = df["label"].to_numpy(dtype=np.float32)
    return TwoTowerDataset(hist_mat, ids, labels), hist_mat


# ============================================================
# 4.双塔模型
# ============================================================
class TwoTower(nn.Module):
    def __init__(
        self,
        n_items,
        emb_dim=EMB_DIM,
        pad_idx=0,
        genre_mat=None,
        pooling=POOLING,
        use_cosine=USE_COSINE,
    ):
        #pad_idx告诉 Embedding 表：第 0 号索引是占位符，请把它钉死成全 0，并且不要训练它
        super().__init__()
        self.pooling = pooling
        self.use_cosine = use_cosine
        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=pad_idx)
        #输入的形状是【 a，b ，c 】，输出就会多一维变成【a，b，c，emb_dim】
        if genre_mat is not None:
            self.register_buffer("genre_mat", torch.tensor(genre_mat))
            self.genre_proj = nn.Linear(genre_mat.shape[1], emb_dim, bias=False)
        else:
            self.genre_mat = None
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )

    def _item_vec(self, idx):
        v = self.item_emb(idx)
        if self.genre_mat is not None:
            v = v + self.genre_proj(self.genre_mat[idx])
        return v

    def forward(self, user_seq, item_idx):
        # 用户塔必须与候选物品无关，否则无法离线预计算用户向量、Faiss 召回就失效了。
        # 所以这里只能用「候选无关」的池化：平均 / 最近 N 个。依赖候选的注意力（DIN）属于精排。
        h = self._item_vec(user_seq)

        if self.pooling == "last":
            # 历史是左补 0，所以最后一个位置就是最近的物品
            pooled = h[:, -1, :]
        elif self.pooling == "mask":
            # 写法不同，但和 mean 数学等价（用来验证 mask 写法没引入 bug）
            m = (user_seq != 0).float().unsqueeze(-1)
            pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        else:
            pooled = h.sum(dim=1) / (user_seq != 0).sum(dim=1, keepdim=True).clamp(min=1)

        user_vec = self.user_mlp(pooled)
        item_vec = self.item_mlp(self._item_vec(item_idx))

        if self.use_cosine:
            user_vec = F.normalize(user_vec, dim=-1)
            item_vec = F.normalize(item_vec, dim=-1)
        return (user_vec, item_vec)

def load_trained_two_tower(path, device):
    """按 checkpoint 中保存的训练配置恢复 TwoTower。"""
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "state_dict" not in checkpoint:
        raise ValueError(
            f"{path} 是旧格式 checkpoint，请用新版 two_tower.py 重新训练"
        )

    config = checkpoint["config"]
    item2idx, n_items = build_vocab()

    if n_items != config["n_items"]:
        raise ValueError(
            f"词表物品数 {n_items} 与 checkpoint {config['n_items']} 不一致"
        )
    if config["max_len"] != MAX_LEN:
        raise ValueError(
            f"checkpoint max_len={config['max_len']}，"
            f"当前代码 MAX_LEN={MAX_LEN}"
        )

    genre_mat = build_genre_matrix(item2idx) if config["use_genres"] else None
    model = TwoTower(
        n_items,
        emb_dim=config["emb_dim"],
        pad_idx=config["pad_idx"],
        genre_mat=genre_mat,
        pooling=config["pooling"],
        use_cosine=config["use_cosine"],
    ).to(device)

    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, config

# ============================================================
# 5. 一个 epoch 的训练循环
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss=0.0
    for seq,item,label in loader:
        seq=seq.to(device)
        item=item.to(device)
        label=label.to(device)
        optimizer.zero_grad()
        user_vec,item_vec=model(seq,item)
        score=(user_vec*item_vec).sum(dim=1,keepdim=True)
        #user_one=F.normalize(user_vec,dim=-1)
        #item_one=F.normalize(item_vec,dim=-1)
        #score=(user_one*item_one).sum(dim=1,keepdim=True)
        #此处用内积计算双塔输出最终结果，可以加个归一化后再内积
        loss=loss_fn(score,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    return total_loss/len(loader)



# ============================================================
# 6. 评估
# ============================================================
@torch.no_grad()
def evaluate_auc(model, loader, device):
    model.eval()
    probs, labels = [], []
    for seq, item, label in loader:
        seq = seq.to(device)
        item = item.to(device)
        user_vec, item_vec = model(seq, item)
        score = (user_vec * item_vec).sum(dim=1, keepdim=True)
        probs.append(torch.sigmoid(score).cpu().numpy().ravel())
        labels.append(label.numpy().ravel())
    return roc_auc_score(np.concatenate(labels), np.concatenate(probs))


@torch.no_grad()
def export_vectors(model, valid_df, item2idx, n_items, device, out_dir=None):
    out_dir = out_dir or EMB_OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    model.eval()

    idxs = torch.arange(1, n_items, device=device)
    chunks = []
    for chunk in torch.split(idxs, 4096):
        dummy_seq = torch.zeros(chunk.shape[0], MAX_LEN, dtype=torch.long, device=device)
        _, item_vec = model(dummy_seq, chunk)
        chunks.append(item_vec.cpu().numpy())
    item_vectors = np.concatenate(chunks).astype("float32")
    np.save(os.path.join(out_dir, "item_vectors.npy"), item_vectors)

    pos = valid_df[valid_df["label"] == 1].reset_index(drop=True)
    hist_mat = build_history_matrix(pos, item2idx)
    user_chunks = []
    for start in range(0, len(pos), 1024):
        seq = torch.tensor(hist_mat[start:start + 1024], dtype=torch.long, device=device)
        dummy_item = torch.ones(seq.shape[0], dtype=torch.long, device=device)
        user_vec, _ = model(seq, dummy_item)
        user_chunks.append(user_vec.cpu().numpy())
    user_vectors = np.concatenate(user_chunks).astype("float32")
    np.save(os.path.join(out_dir, "valid_user_vectors.npy"), user_vectors)

    targets = pos["target_item_id"].map(lambda x: item2idx.get(int(x), -1)).to_numpy()
    np.save(os.path.join(out_dir, "valid_targets.npy"), targets)

    print(f"  已导出：物品向量 {item_vectors.shape}，用户向量 {user_vectors.shape}")
    print(f"  目录：{out_dir}")


# ============================================================
# 7. 主流程
# ============================================================
def main():
    global SEED
    args = parse_args()
    SEED = args.seed
    set_seed(SEED)

    out_dir = (
        args.out_dir
        if os.path.isabs(args.out_dir)
        else os.path.join(_HERE, args.out_dir)
    )
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"设备：{device}")
    print(
        f"配置：pooling={args.pooling} genres={args.genres} "
        f"cosine={args.cosine} epochs={args.epochs} seed={args.seed}"
    )

    print("读数据 ...")
    train_df = load_split("train", MAX_ROWS)
    valid_df = load_split("valid")
    test_df = load_split("test")
    print(f"  train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)}")

    item2idx, n_items = build_vocab()
    genre_mat = build_genre_matrix(item2idx) if args.genres else None
    print(f"  物品数（含 padding 位）：{n_items}")

    print("构造输入 ...")
    train_set, train_hist = build_dataset(train_df, item2idx)
    validate_history_matrix(train_df, train_hist, item2idx)
    valid_set, _ = build_dataset(valid_df, item2idx)
    test_set, _ = build_dataset(test_df, item2idx)
    print(f"  history 矩阵形状：{train_hist.shape}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=4096, shuffle=False)

    model = TwoTower(
        n_items,
        genre_mat=genre_mat,
        pooling=args.pooling,
        use_cosine=args.cosine,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        auc = evaluate_auc(model, valid_loader, device)
        print(f"epoch {epoch}  train_loss={loss:.4f}  valid_AUC={auc:.4f}")

    print()
    test_auc = evaluate_auc(model, test_loader, device)
    print(f"test AUC = {test_auc:.4f}")

    export_vectors(model, valid_df, item2idx, n_items, device, out_dir=out_dir)

    # 保存召回模型本身：build_hard_negatives.py 要用它给训练集算用户向量
    ckpt_path = os.path.join(out_dir, "model_two_tower.pt")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                "n_items": n_items,
                "emb_dim": EMB_DIM,
                "pad_idx": 0,
                "use_genres": args.genres,
                "pooling": args.pooling,
                "use_cosine": args.cosine,
                "max_len": MAX_LEN,
            },
        },
        ckpt_path,
    )
    print(f"  召回模型已保存：{ckpt_path}")

if __name__ == "__main__":
    main()
