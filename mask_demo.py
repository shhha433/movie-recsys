"""
masked mean 演示：用最小的数字，把每一步都打印出来
=====================================================
跑法：
    /opt/miniconda3/bin/python outputs/week3/mask_demo.py

建议你改几个数字再跑一次，比如：
    - 把 user_seq 改成自己的数据
    - 把 PAD_ROW 改成别的值，看结果怎么被污染
    - 把 EMB 里的向量改掉
"""

import torch
import torch.nn as nn

# ============================================================
# 0. 设定一个最小的例子：2 个用户、最多 4 个历史、每个物品 2 维
# ============================================================
MAX_LEN = 4
EMB_DIM = 2

# 历史序列（0 = padding）
user_seq = torch.tensor([
    [0, 0, 2, 3],      # 用户 A：真实历史 = 物品 2、物品 3（前面 2 个位置是空的）
    [0, 5, 4, 1],      # 用户 B：真实历史 = 物品 5、物品 4、物品 1（只有 1 个空位）
])

# embedding 表：第 i 行是物品 i 的向量（编几个好算的数）
EMB = torch.tensor([
    [0.0, 0.0],        # 索引 0 = padding
    [1.0, 0.0],        # 物品 1
    [0.0, 1.0],        # 物品 2
    [1.0, 1.0],        # 物品 3
    [2.0, 0.0],        # 物品 4
    [0.0, 2.0],        # 物品 5
])


def show(name, t):
    print(f"{name:<28} shape={tuple(t.shape)}  = {t.tolist()}")


print("=" * 64)
print("输入")
print("=" * 64)
show("user_seq", user_seq)
show("EMB（embedding 表）", EMB)
print()
print("手算的正确答案（只看非 0 位置）：")
print("  用户 A：2 个物品 → (物品2 + 物品3) / 2 = ([0,1] + [1,1]) / 2 = [0.5, 1.0]")
print("  用户 B：3 个物品 → (物品5 + 物品4 + 物品1) / 3 = ([0,2]+[2,0]+[1,0]) / 3 = [1.0, 0.667]")
print()

# ============================================================
# 1. 第一步：查 embedding —— 注意这里出来的已经是三维
# ============================================================
emb = EMB[user_seq]                       # 等价于 nn.Embedding 的前向
print("=" * 64)
print("第 1 步：查 embedding")
print("=" * 64)
show("emb = EMB[user_seq]", emb)
print("  ↑ 形状从 [B, L] = [2, 4] 变成了 [B, L, d] = [2, 4, 2]")
print("    多出来的最后一维 d=2 是'每个物品的向量'")
print()

# ============================================================
# 2. 第二步：造 mask
# ============================================================
mask_bool = (user_seq != 0)
mask = mask_bool.float().unsqueeze(-1)
print("=" * 64)
print("第 2 步：造 mask")
print("=" * 64)
show("user_seq != 0", mask_bool)
print("  ↑ 逐元素判断：这个位置有没有真实物品")
print()
show("(user_seq != 0).float()", mask_bool.float())
print("  ↑ True/False 变成 1.0/0.0")
print()
show("mask = ....unsqueeze(-1)", mask)
print("  ↑ 关键的一步：形状从 [B, L] = [2, 4] 变成 [B, L, 1] = [2, 4, 1]")
print("    为什么要加这一维？因为要和 emb 的 [B, L, d] 对齐才能相乘")
print()

# ============================================================
# 3. 第三步：相乘，把 padding 位置的向量清零
# ============================================================
masked_emb = emb * mask
print("=" * 64)
print("第 3 步：emb * mask —— padding 位置被清零")
print("=" * 64)
show("emb * mask", masked_emb)
print("  ↑ 对比上面的 emb：用户 A 的前两行、用户 B 的第 1 行都变成 [0,0] 了")
print()

# ============================================================
# 4. 第四步：求和 + 除以真实个数
# ============================================================
summed = masked_emb.sum(dim=1)
count = mask.sum(dim=1).clamp(min=1)
user_vec = summed / count
print("=" * 64)
print("第 4 步：求和 ÷ 真实个数")
print("=" * 64)
show("masked_emb.sum(dim=1)", summed)
show("mask.sum(dim=1)", count)
show("user_vec = summed / count", user_vec)
print()
print("  ✅ 和手算的正确答案一致：[[0.5, 1.0], [1.0, 0.667]]")
print()

# ============================================================
# 5. 反例一：直接用 sum（不做 mask，也不除个数）
# ============================================================
print("=" * 64)
print("反例 1：如果直接 emb.sum(dim=1)（既不 mask 也不除个数）")
print("=" * 64)
show("emb.sum(dim=1)", emb.sum(dim=1))
print("  ❌ 用户 A 有 2 个物品、用户 B 有 3 个物品，光看和没法比：")
print("     B 的 [3,2] 看着比 A 的 [1,2] 大，但这只是因为 B 看得多，不代表兴趣更强")
print("     （如果所有用户历史长度都一样，除以个数只是个常数缩放，问题还不大；")
print("       真实数据里长度千差万别，所以必须除以各自的真实个数）")
print()

# ============================================================
# 6. 反例二：没有 padding_idx 时，第 0 行不是 0 会怎样
# ============================================================
print("=" * 64)
print("反例 2：如果没有 padding_idx=0（第 0 行被训练成非零）")
print("=" * 64)
EMB_BAD = EMB.clone()
EMB_BAD[0] = torch.tensor([9.0, 9.0])       # 假设训练把它变成了 [9,9]
emb_bad = EMB_BAD[user_seq]
show("emb_bad[0]（用户 A 的 embedding）", emb_bad[0])
sum_bad = emb_bad.sum(dim=1)
res_bad = sum_bad / mask.sum(dim=1).clamp(min=1)
show("用户 A 的均值", res_bad[0])
show("用户 B 的均值", res_bad[1])
print()
print("  手算：用户 A = ([9,9] + [9,9] + [0,1] + [1,1]) / 2 = [19,20]/2 = [9.5, 10.0]")
print("  ❌ 正确答案应该是 [0.5, 1.0]，现在偏了 19 倍")
print("  ❌ 而且用户 A 有 2 个 padding、用户 B 只有 1 个，污染程度不一样")
print("     —— 用户向量之间被引入了'和真实历史长度相关'的系统性偏差")
print()

# ============================================================
# 7. 不写 unsqueeze(-1) 会怎样
# ============================================================
print("=" * 64)
print("反例 3：mask 不写 .unsqueeze(-1) 会怎样")
print("=" * 64)
try:
    wrong = emb * mask_bool.float()
    print("没报错，结果形状 =", tuple(wrong.shape))
    print("（如果不报错，说明维度凑巧对上了，但语义完全是错的）")
except RuntimeError as e:
    print(f"报错：{e}")
    print()
    print("  [B, L, d] 和 [B, L] 右对齐后变成 [1, B, L]，中间维度对不上，所以报错")
    print("  加 .unsqueeze(-1) 变成 [B, L, 1] 就能正确广播了")
