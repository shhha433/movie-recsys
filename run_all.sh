#!/bin/bash

cd "$(dirname "$0")" || exit 1

PY=/opt/miniconda3/bin/python
if [ ! -x "$PY" ]; then
    echo "找不到 $PY，请先确认 Python 环境。"
    exit 1
fi

step() {
    local title="$1"
    local script="$2"
    shift 2

    echo
    echo "============================================================"
    echo "  $title"
    echo "============================================================"

    "$PY" "$script" "$@" || {
        echo "第 $title 步失败，已中止。"
        exit 1
    }
}

TOWER_OUT="${TOWER_OUT:-emb_mask_genre}"
TOWER_POOLING="${TOWER_POOLING:-mask}"
TOWER_GENRES="${TOWER_GENRES:-1}"
TOWER_CKPT="$TOWER_OUT/model_two_tower.pt"

if [ "$TOWER_GENRES" = "1" ]; then
    TOWER_GENRE_FLAG=(--genres)
else
    TOWER_GENRE_FLAG=(--no-genres)
fi

step "1/6  数据处理" prepare_data.py

step "2/6  训练召回层双塔" two_tower.py \
    --pooling "$TOWER_POOLING" \
    "${TOWER_GENRE_FLAG[@]}" \
    --out-dir "$TOWER_OUT"

step "3/6  召回检索评估" recall_eval.py \
    --experiments "$TOWER_OUT" \
    --baseline "$TOWER_OUT"
step "4/6  从召回候选挖难负样本" build_hard_negatives.py \
  --ckpt "$TOWER_CKPT" \
  --out "data/processed/train_hard_${TOWER_OUT}.csv"

step "5/6  训练排序层 DIN" din.py \
  --train-split "train_hard_${TOWER_OUT}" \
  --ckpt-out "$TOWER_OUT/model_din.pt"

step "6/6  召回→排序 端到端评估" rank_eval.py \
  --emb-dir "$TOWER_OUT" \
  --ckpt "$TOWER_OUT/model_din.pt"

echo "全部跑完。"
echo "本次召回目录：$TOWER_OUT"
echo "本次召回模型：$TOWER_CKPT"
read -r -p "按回车键关闭这个窗口..."
