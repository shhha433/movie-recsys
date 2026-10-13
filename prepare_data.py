"""
第 2 周任务 3:MovieLens-1M 数据处理
================================
把原始的评分日志，变成模型能直接吃的训练样本表。

你只需要填 2 个 TODO:
    TODO-1  按时间切分 train / valid / test
    TODO-2  构造样本（历史序列 + 正样本 + 随机负采样）

其余部分（读文件、打印统计、自动自检）都不用改。

运行：
    /opt/miniconda3/bin/python outputs/week2/prepare_data.py

MovieLens-1M 原始格式（用 :: 分隔）：
    ratings.dat   UserID::MovieID::Rating::Timestamp
                  1::1193::5::978300760     ← 用户 1 给电影 1193 打了 5 分
                  1::661::3::978302109      ← 打 3 分
    movies.dat    MovieID::Title::Genres
                  1::Toy Story (1995)::Animation|Children's|Comedy

和 MIND 的对应关系（以后要换成 MIND 时对照着看）：
    MIND 的 user_id        ↔ MovieLens 的 UserID
    MIND 的 time           ↔ MovieLens 的 Timestamp
    MIND 的 history        ↔ 该用户之前评分过的电影（由你在 TODO-2 构造）
    MIND 的「点击了」       ↔ MovieLens 里 Rating >= 4 的交互（当作「喜欢」）
"""

import os
import sys
import random

import pandas as pd

# ============================================================
# 配置
# ============================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_HERE, "data", "ml-1m")
OUT_DIR = os.path.join(_HERE, "data", "processed")

RATINGS_PATH = os.path.join(DATA_DIR, "ratings.dat")
MOVIES_PATH = os.path.join(DATA_DIR, "movies.dat")

MIN_RATING = 4        # 评分 >= 4 才算「喜欢」，拿来当正样本
MAX_HISTORY = 3      # 历史序列最多保留最近多少条
NEG_PER_POS = 4       # 每个正样本配几个随机负样本（1 表示正负比 1:1）
MAX_USERS = 0         # 0 = 全部用户；调试时可以先设成 500
SPLIT = (0.8, 0.1, 0.1)
SEED = 42
random.seed(SEED)



# ============================================================
# 0. 先检查数据在不在
# ============================================================
def check_data():
    missing = [p for p in (RATINGS_PATH, MOVIES_PATH) if not os.path.exists(p)]
    if not missing:
        return True
    print("找不到这些文件：")
    for p in missing:
        print("   " + p)
    print()
    print("请把 MovieLens-1M 解压到：")
    print("   " + DATA_DIR)
    print("目录里应该有 ratings.dat / movies.dat / users.dat 三个文件。")
    print("下载:https://grouplens.org/datasets/movielens/1m/")
    return False


# ============================================================
# 1. 读原始数据（不用改）
# ============================================================
def load_ratings(path, min_rating=MIN_RATING, max_users=MAX_USERS):
    """ratings.dat → DataFrame[user_id, item_id, rating,time]
    只保留评分 >= min_rating 的交互，这些当作「用户喜欢」。
    """
    rows = []
    total = 0
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
#with open(...)：打开文件，with 保证用完自动关闭。
#encoding="latin-1"：MovieLens 老数据集的编码，用 UTF-8 可能报错，所以用 latin-1。
#or line in fh：一行一行读。
#ine.strip()：去掉行首行尾的空白和换行符。
#if not line: continue：空行跳过。
            parts = line.split("::")
            if len(parts) != 4:
                continue
#数据是一个四列的列表，用 split("::") 分开，长度不等于 4 的行就跳过。
            total += 1
            rating = int(parts[2])
            if rating < min_rating:
                continue
#每得到一行数据total就加一。把rating改成整数。只保留评分 >= min_rating 的交互，这些当作「用户喜欢」。
            rows.append({
                "user_id": int(parts[0]),
                "item_id": int(parts[1]),
                "time": int(parts[3]),
                "rating": rating,
            })
    df = pd.DataFrame(rows)
    print(f"  原始评分 {total} 条，保留评分 >= {min_rating} 的 {len(df)} 条（{len(df) / total:.0%}）")
    if max_users:
        keep = sorted(df["user_id"].unique())[:max_users]
        df = df[df["user_id"].isin(keep)].copy()
        print(f"  【调试模式】只保留前 {max_users} 个用户，剩 {len(df)} 条交互")
    return df


def bad_data(path, max_users=MAX_USERS):
    """ratings.dat → DataFrame[user_id, item_id, rating,time]
    只保留评分 >= min_rating 的交互，这些当作「用户喜欢」。
    """
    rows = []
    total = 0
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) != 4:
                continue
            total += 1
            rating = int(parts[2])
            if rating > 2:
                continue
            rows.append({
                "user_id": int(parts[0]),
                "item_id": int(parts[1]),
                "time": int(parts[3]),
                "rating": rating,
            })
    bd= pd.DataFrame(rows)
    if max_users:
        keep = sorted(bd["user_id"].unique())[:max_users]
        bd = bd[bd["user_id"].isin(keep)].copy()
    return bd
#if max_users:：如果 max_users 不是 None、不是 0（即真值），就执行。
#df["user_id"].unique()：取出所有不重复的用户 id。
#sorted(...)[:max_users]：排序后取前 max_users 个用户。
#df["user_id"].isin(keep)：筛选出这些用户的行。
#.copy()：复制一份，避免后面改动影响原表（避免 pandas 的 SettingWithCopyWarning）。
#作用：调试时只跑前 N 个用户的数据，比如 max_users=100，几秒就能跑完，方便开发。正式跑时把 max_users 设成 None 或 0 就处理全量。


def load_movies(path):
    rows = []
    with open(path, encoding="latin-1") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split("::")
            if len(parts) < 3:
                continue
            rows.append({"item_id": int(parts[0]), "title": parts[1], "genres": parts[2]})
    return pd.DataFrame(rows)


# ============================================================
# 2. TODO-1：按时间切分
# ============================================================
def split_by_time(df, ratios=SPLIT):
    """
    把所有交互按**时间先后**切成 train / valid / test,返回三个 DataFrame。

    为什么不能随机切分？（第 1 周思考题 4 讲过）
        随机切分会让训练集里混进「未来」的行为，测试变成开卷考试，指标虚高。
        真实系统必须用过去预测未来。

    提示（三步）：
        1) 先按时间排序
        2) 算三个切分点：
        3) 用切片返回三份：
    """
    # TODO-1: 删掉下面这行，写你的代码
    df=df.sort_values("time")
    n=len(df)
    n_train=int(n*ratios[0])
    n_valid=int(n*(ratios[0] + ratios[1]))
    return df.iloc[:n_train], df.iloc[n_train:n_valid], df.iloc[n_valid:]


# ============================================================
# 3. TODO-2：构造样本 + 随机负采样
# ============================================================
def build_samples(part_df,bd, user_seen, max_history=MAX_HISTORY, neg_per_pos=NEG_PER_POS):
    """
    把交互日志变成一张训练样本表，列是：
        user_id | history | target_item_id | label

    每一行代表一个「预测任务」：
        给定这个用户的历史序列 history,预测他会不会喜欢 target_item_id。

    正样本：用户真实评分 >= 4 的电影
    负样本：从全库随机抽一个「他没看过的」电影，标 0

    例子（某个用户的观影序列，按时间排序 = [M1, M2, M3, M4],
          max_history = 2,NEG_PER_POS = 1):
            历史 "M1"     → 目标 M2    label 1   ← 正样本
            历史 "M1"     → 目标 M77   label 0   ← 随机负样本
            历史 "M1 M2"  → 目标 M3    label 1
            历史 "M1 M2"  → 目标 M15   label 0
            历史 "M2 M3"  → 目标 M4    label 1
            历史 "M2 M3"  → 目标 M88   label 0
    提示（四步）：
        1) 按用户分组，组内按时间排序：
               for user_id, g in part_df.groupby("user_id", sort=False):
                   g = g.sort_values("time")
                   items = g["item_id"].tolist()
        2) 从第 2 个开始滑窗（第 1 个没有历史，做不了样本）：
               for i in range(1, len(items)):
        3) 历史 = 前面最近的 max_history 个：
               hist = items[max(0, i - max_history):i]
           写一行正样本：[user_id, " ".join(map(str, hist)), items[i], 1]
        4) 再配 neg_per_pos 个负样本：
               从 all_items 里 random.choice 抽一个；
               抽到「在 user_seen[user_id] 里」或「等于当前目标」的就重抽(while)
               写一行：[user_id, " ".join(map(str, hist)), neg, 0]

    参数：
        part_df   切分后的某一份交互(train / valid / test)
        all_items 全部电影 ID 的列表（用来抽负样本）
        user_seen dict:user_id → 该用户在**全量数据**里看过的电影集合

    返回:pd.DataFrame,列名必须是 ["user_id", "history", "target_item_id", "label"]
        history 存成字符串（" ".join(...)），方便写文件
    """
    # TODO-2: 删掉下面这行，写你的代码
    samples = []
    for user_id,g in part_df.groupby("user_id",sort=False):
        g=g.sort_values("time")
        items=g["item_id"].tolist()
        for i in range(1,len(items)):
            hist=items[max(0,i-max_history):i]
            samples.append((user_id, " ".join(map(str, hist)), items[i], 1))
            for _ in range(neg_per_pos):
                neg=random.choice(bd)
                while neg in user_seen[user_id] or neg==items[i]:
                    neg=random.choice(bd)
                samples.append((user_id, " ".join(map(str, hist)), neg, 0))
    return pd.DataFrame(samples, columns=["user_id", "history", "target_item_id", "label"])



# ============================================================
# 4. 统计与自动自检（不用改）
# ============================================================
def describe(name, samples):
    n = len(samples)
    if n == 0:
        print(f"  {name:<5} 0 条样本")
        return
    pos = int((samples["label"] == 1).sum())
    print(f"  {name:<5} {n:>8} 条样本   正样本 {pos:>8} ({pos / n:.0%})   负样本 {n - pos:>8}")


def auto_check(train_df, valid_df, test_df, samples, item_sets, user_seen):
    problems = []

    def fail(msg):
        problems.append(msg)

    for name, d in (("train", train_df), ("valid", valid_df), ("test", test_df)):
        if d is None or len(d) == 0:
            fail(f"{name} 的交互是空的")

    if len(train_df) and len(valid_df) and train_df["time"].max() > valid_df["time"].min():
        fail("train 里出现了比 valid 更晚的数据 —— 时间切分错了（必须先按 time 排序再切）")
    if len(valid_df) and len(test_df) and valid_df["time"].max() > test_df["time"].min():
        fail("valid 里出现了比 test 更晚的数据")

    for name in ("train", "valid", "test"):
        s = samples[name]
        if s is None or len(s) == 0:
            fail(f"{name} 的样本表是空的")
            continue
        if list(s.columns) != ["user_id", "history", "target_item_id", "label"]:
            fail(f"{name} 的列名不对，应为 ['user_id','history','target_item_id','label']，实际是 {list(s.columns)}")
            continue
        if not set(s["label"].unique()) <= {0, 1}:
            fail(f"{name} 的 label 只能取 0 或 1")
        too_long = int((s["history"].str.split().str.len() > MAX_HISTORY).sum())
        if too_long:
            fail(f"{name} 里有 {too_long} 条历史序列长度超过 {MAX_HISTORY}")
        leak = 0
        for r in s.itertuples(index=False):
            if r.label == 0 and r.target_item_id in user_seen.get(r.user_id, set()):
                leak += 1
        if leak:
            fail(f"{name} 里有 {leak} 条负样本其实是用户看过的电影（负样本必须没看过）")
        bad_pos = int(((s["label"] == 1) & (~s["target_item_id"].isin(item_sets[name]))).sum())
        if bad_pos:
            fail(f"{name} 里有 {bad_pos} 条正样本的目标不在该份数据里")

    print("自检结果：")
    if problems:
        for p in problems:
            print("  [问题] " + p)
        print()
        print("先修上面这些问题，再看下面的统计。")
    else:
        print("  全部通过 ✅")
    return not problems


def main():
    if not check_data():
        sys.exit(1)

    print("读数据 ...")
    bd = bad_data(RATINGS_PATH)["item_id"].tolist()
    movies = load_movies(MOVIES_PATH)
    ratings = load_ratings(RATINGS_PATH)
    user_seen = ratings.groupby("user_id")["item_id"].apply(set).to_dict()
    print(f"  电影 {len(movies)} 部，交互 {len(ratings)} 条，用户 {ratings['user_id'].nunique()} 个")
    print()
    print("原始交互长这样（前 3 条）：")
    print(ratings.head(3).to_string(index=False))
    print()

    train_df, valid_df, test_df = split_by_time(ratings)
    print(f"按时间切分：train {len(train_df)} / valid {len(valid_df)} / test {len(test_df)} 条交互")
    print(f"  train 时间 {train_df['time'].min()} ~ {train_df['time'].max()}")
    print(f"  valid 时间 {valid_df['time'].min()} ~ {valid_df['time'].max()}")
    print(f"  test  时间 {test_df['time'].min()} ~ {test_df['time'].max()}")
    print()

    item_sets = {
        "train": set(train_df["item_id"]),
        "valid": set(valid_df["item_id"]),
        "test": set(test_df["item_id"]),
    }

    print("构造样本中（要跑一会儿）...")
    train_s = build_samples(train_df, bd, user_seen)
    print("  train 完成")
    valid_s = build_samples(valid_df, bd, user_seen)
    print("  valid 完成")
    test_s = build_samples(test_df, bd, user_seen)
    print("  test 完成")
    print()

    samples = {"train": train_s, "valid": valid_s, "test": test_s}
    ok = auto_check(train_df, valid_df, test_df, samples, item_sets, user_seen)

    print()
    print("样本统计：")
    describe("train", train_s)
    describe("valid", valid_s)
    describe("test", test_s)
    print()
    print("训练样本长这样（前 5 行）：")
    print(train_s.head(5).to_string(max_colwidth=40))

    if ok:
        os.makedirs(OUT_DIR, exist_ok=True)
        print()
        for name, s in (("train", train_s), ("valid", valid_s), ("test", test_s)):
            path = os.path.join(OUT_DIR, f"{name}.csv")
            s.to_csv(path, index=False)
            print(f"  已写出 {path}（{os.path.getsize(path) / 1024 / 1024:.1f} MB）")
        print()
        print("把「自检结果」和「样本统计」这几行发给 Codex。")


if __name__ == "__main__":
    main()
