"""
第 3 周主线：双塔召回模型
================================
把第 2 周产出的样本表（227 万行）喂给一个双塔模型，训练完再用 Faiss 算 Recall@50。

这是项目 A 的骨架，也是简历上第一条能写的东西。

你要填 3 个 TODO：
    TODO-1  build_history_matrix   把 history 字符串变成定长整数矩阵
    TODO-2  TwoTower.forward       两个塔 + 相似度打分
    TODO-3  train_one_epoch        一个 epoch 的训练循环（week1 写过，这次自己搭）

其余部分（读数据、建词表、Dataset、AUC 评估、Faiss 检索）都不用改。


调试建议：先把 MAX_ROWS 设成 200000 跑通，再改成 0（用全量）。
"""

import os
import sys
import random

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
_HERE = os.path.dirname(os.path.abspath(__file__))          # wo/outputs/week3                                   # wo
DATA_DIR = os.path.join(_HERE, "data","processed")                # wo/outputs/week2/data
MOVIES_PATH = os.path.join(_HERE, "data", "ml-1m", "movies.dat")

MAX_LEN = 10          # history 最多保留最近多少个物品
EMB_DIM = 32          # 每层 embedding 的维度
BATCH_SIZE = 2048
EPOCHS = 3
LR = 1e-3
# ⚠️ 跑对照实验时，只改这一行：把 "emb" 换成实验名，例如 "emb_baseline" / "emb_lastitem"
#    目录会自动创建，不会覆盖别的实验。
EMB_OUT_DIR = os.path.join(_HERE, "emb_baseline_full")
SEED = 42
MAX_ROWS = 0      # 0 = 用全部；调试时设成 200000


def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ============================================================
# 1. 读数据与建词表（不用改）
# ============================================================
def load_split(name, max_rows=None):
    """读某一份数据。max_rows 只用来缩小【训练集】做快速调试——
    验证集和测试集永远用全量，否则不同实验之间的指标没法比较。"""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"找不到 {path}")
        print("先跑 outputs/week2/prepare_data.py 生成三个 csv。")
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


# ============================================================
# 2. TODO-1：把 history 字符串转成定长整数矩阵
# ============================================================
def build_history_matrix(df, item2idx, max_len=MAX_LEN):
    """
    把 df 的 "history" 列（形如 "858 2384 593" 的字符串）转成形状 [n, max_len] 的整数矩阵。

    要求：
        1. 每行取该用户**最近的** max_len 个物品；
           如果不足 max_len 个，**缺的部分补 0**（0 是 padding）
        2. 物品 id 用 item2idx 映射成连续索引
        3. 返回 numpy 数组，形状 [len(df), max_len]，整数类型

    例子（max_len = 4）：
        history = "11 22 33"      →  [0, 11', 22', 33']      ← 前面补一个 0
        history = "11 22 33 44 55" →  [22', 33', 44', 55']   ← 只留最近 4 个
        history = "11"            →  [0, 0, 0, 11']
        history = ""              →  [0, 0, 0, 0]
        （11' 表示 item2idx[11]；映射不到的物品也记 0）

    验收：矩阵形状正确、非零元素个数 ≤ 对应行的 history 长度、每行真实物品数量不超过 max_len
    """
    # TODO-1: 删掉下面这行，写你的代码
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


def check_history_matrix(df, mat, item2idx, max_len=MAX_LEN):
    """自动检查 TODO-1 写对了没有。"""
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
        print("TODO-1 自检：")
        for p in problems:
            print("  [问题] " + p)
    return problems


# ============================================================
# 3. Dataset（不用改）
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
# 4. TODO-2：双塔模型
# ============================================================
class TwoTower(nn.Module):
    def __init__(self, n_items, emb_dim=EMB_DIM, pad_idx=0):
        #pad_idx告诉 Embedding 表：第 0 号索引是占位符，请把它钉死成全 0，并且不要训练它
        super().__init__()
        self.pad_idx = pad_idx
        self.item_emb = nn.Embedding(n_items, emb_dim, padding_idx=pad_idx)
        #输入的形状是【 a，b ，c 】，输出就会多一维变成【a，b，c，emb_dim】
        self.user_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )
        self.item_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.ReLU(), nn.Linear(emb_dim, emb_dim)
        )

    def forward(self, user_seq, item_idx):
        """
        user_seq: [B, max_len] 的历史物品索引（0 是 padding）
        item_idx: [B]          的候选物品索引

        要求：
            1. 用户塔：把 user_seq 里每个物品查 embedding，
               然后对**非 padding 的位置**求平均（padding 不能算进去！），
               再过 self.user_mlp，得到 [B, emb_dim]
            2. 物品塔：把 item_idx 查 embedding，过 self.item_mlp，
               得到 [B, emb_dim]
            3. 返回两个向量 (user_vec, item_vec)，**不用**算相似度

        提示：只用一层 nn.Embedding 就行（历史物品和目标物品共用同一张表）
        验收：两个返回值形状都是 [B, emb_dim]；padding 的位置不参与平均
        """
        # TODO-2: 删掉下面这行，写你的代码
        user_emd=self.item_emb(user_seq)
        user_vec=self.user_mlp(user_emd.sum(dim=1)/((user_seq!=0).sum(dim=1,keepdim=True).clamp(min=1)))
        #此处求平均是因为用户塔输出数据的规模是确定的，但用户历史在变长，要求平均去压缩，还有就是默认每个物品权重一致，也可以选择din、最近、时间衰减加权等
        #均值另一种算法如下
        #mask=(user_seq!=0).float().unsqueeze(-1)
        #masked_emd=user_emd*mask
        #user_vec=self.user_mlp(masked_emd.sum(dim=1)/mask.sum(dim=1).clamp(min=1))
        #user_vec=self.user_mlp(user_emd[:,-1,:])
        #取最近一个序列，因为是左补0，所以这样做，右就需要算非0的长度减一位置为最近一个的索引
        item_vec=self.item_mlp(self.item_emb(item_idx))
        #下面是把数据归一化，这样使得后面计算结果时用的是余弦
        #user_vec=F.normalize(user_vec,dim=-1)
        #item_vec=F.normalize(item_vec,dim=-1)
        return (user_vec,item_vec)


# ============================================================
# 5. TODO-3：一个 epoch 的训练循环
# ============================================================
def train_one_epoch(model, loader, optimizer, loss_fn, device):
    """
    跑完一整个 epoch，返回平均训练 loss（float）。

    要求：就是 week1 那四步（清梯度 → 前向 → 算损失 → 反向 + 更新），
          只不过外面多套一层「遍历 loader」。

    注意几个容易漏的点：
        - model.train() 放在循环最前面
        - 数据要 .to(device)
        - loss 要用 loss.item() 累加（返回普通 float，不是张量）
    验收：能返回一个浮点数；跑几个 epoch，验证集 AUC 应该明显高于 0.5
    """
    # TODO-3: 删掉下面这行，写你的代码

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
# 6. 评估（不用改）
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
def export_vectors(model, valid_df, item2idx, n_items, device, out_dir=EMB_OUT_DIR):
    """把物品塔和用户塔的输出导出成 npy，交给 recall_eval.py 去检索。

    为什么要分成两个文件？因为 macOS 上 torch 和 faiss-cpu 各带一份 OpenMP，
    同一个进程里先后使用会直接崩（segfault）。训练进程导出向量、检索进程只做检索，
    这本来就是工业界的做法（训练和检索服务分离）。
    """
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
# 7. 主流程（不用改）
# ============================================================
def main():
    set_seed()
    device = get_device()
    print(f"设备：{device}")

    print("读数据 ...")
    train_df = load_split("train", MAX_ROWS)
    valid_df = load_split("valid")
    test_df = load_split("test")
    print(f"  train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)}")

    item2idx, n_items = build_vocab()
    print(f"  物品数（含 padding 位）：{n_items}")

    print("构造输入 ...")
    train_set, train_hist = build_dataset(train_df, item2idx)
    check_history_matrix(train_df, train_hist, item2idx)
    valid_set, _ = build_dataset(valid_df, item2idx)
    test_set, _ = build_dataset(test_df, item2idx)
    print(f"  history 矩阵形状：{train_hist.shape}")

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    valid_loader = DataLoader(valid_set, batch_size=4096, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=4096, shuffle=False)

    model = TwoTower(n_items).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()
    print(f"模型参数量：{sum(p.numel() for p in model.parameters()):,}")
    print()

    for epoch in range(EPOCHS):
        loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        auc = evaluate_auc(model, valid_loader, device)
        print(f"epoch {epoch}  train_loss={loss:.4f}  valid_AUC={auc:.4f}")

    print()
    test_auc = evaluate_auc(model, test_loader, device)
    print(f"test AUC = {test_auc:.4f}")

    print()
    export_vectors(model, valid_df, item2idx, n_items, device)
    print()
    print("接着跑这一条，用 Faiss 算 Recall@50：")
    print("    /opt/miniconda3/bin/python outputs/week3/recall_eval.py")
    print("把两次运行的输出一起发给 Codex。")


if __name__ == "__main__":
    main()
