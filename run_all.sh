#!/bin/bash
#
# 一键按顺序跑完整条链路：数据处理 → 召回训练 → 召回评估 → 排序训练 → 端到端评估
#
# 用法：在终端里 ./run_all.sh，或双击本文件
# 耗时：全量数据在 CPU 上大约 30-40 分钟（din.py 一步就要 6-8 分钟）
#
# 为什么要拆成 5 个进程：macOS 上 faiss-cpu 和 PyTorch 各带一份 OpenMP，
# 同一个进程里先后使用会静默崩溃（退出码 134/139）。拆进程是必需的，不是风格问题。

cd "$(dirname "$0")" || exit 1

PY=/opt/miniconda3/bin/python
if [ ! -x "$PY" ]; then
    echo "找不到 $PY，请先确认 Python 环境。"
    exit 1
fi

step() {
    echo
    echo "============================================================"
    echo "  $1"
    echo "============================================================"
    "$PY" "$2" || { echo "第 $1 步失败，已中止。"; exit 1; }
}

step "1/6  数据处理"              prepare_data.py
step "2/6  训练召回层双塔"          two_tower.py
step "3/6  召回检索评估"            recall_eval.py
step "4/6  从召回候选挖难负样本"    build_hard_negatives.py
step "5/6  训练排序层 DIN"          din.py
step "6/6  召回→排序 端到端评估"    rank_eval.py

echo
echo "全部跑完。"
echo "注意：rank_eval.py 用的是 EMB_DIR（默认 emb_cosineB2_full），"
echo "      而第 2 步只重训 two_tower.py 里 EMB_OUT_DIR 指向的那一份，两者要自己对齐。"
read -r -p "按回车键关闭这个窗口..."
