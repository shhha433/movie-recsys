# 项目代码结构与实验说明

## 数据流

```text
MovieLens-1M ratings/movies
        ↓ prepare_data.py
train / valid / test 样本
        ↓ two_tower.py
用户向量、物品向量、双塔 checkpoint
        ↓ recall_eval.py
Faiss top-50、Recall@50、配对 bootstrap
        ↓ build_hard_negatives.py
条件难负样本
        ↓ din.py
DIN checkpoint
        ↓ rank_eval.py
HR@10 / NDCG@10 / MRR
```

## 文件职责

- `prepare_data.py`：评分清洗、时间切分、滑窗样本、低分随机负样本、数据校验。
- `two_tower.py`：双塔模型，用户历史 mask 池化，物品 ID + genres 表示，向量导出和 checkpoint 配置。
- `recall_eval.py`：Faiss 全库 top-50、Recall@50、bootstrap CI、配对 bootstrap。
- `build_hard_negatives.py`：按召回候选生成条件难负样本，并排除目标、当前历史和此前全部喜欢物品。
- `din.py`：DIN 注意力排序、mask 校验、训练与 checkpoint。
- `rank_eval.py`：在召回 top-50 内用 DIN 重排，计算端到端 HR/NDCG/MRR。
- `examples/mask_demo.py`：padding mask 的最小示例。
- `hot_ratio.py`：召回结果热门度分析。
- `run_all.sh`：串联完整六步流水线。

## 最终实验

固定数据、历史长度 20、mask 池化、10 轮训练和随机种子 42：

| 实验 | Recall@50 | DIN test AUC | HR@10 | NDCG@10 | MRR |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无 genres | 12.12% | 0.8108 | 5.49% | 2.71% | 2.23% |
| genres | **15.19%** | **0.8024** | **6.30%** | **2.99%** | **2.47%** |

genres 的 Recall@50 配对提升为 +3.07pp，95% CI [+2.80, +3.37]pp，p<0.001。

修复后的两份 hard 数据各包含 1,819,384 个负样本，target、history 和历史喜欢物品泄漏均为 0。

## 关键设计

- 用户塔不能看到候选，否则无法离线建索引用 Faiss 召回。
- 历史物品和候选物品必须共用同一个 `_item_vec()`，否则双塔表示空间不一致。
- DIN 只在目标被召回时使用 top-50 难负样本；否则退回全库未喜欢物品。
- 负样本必须排除当前目标和用户此前喜欢过的全部物品。
- 排序只能在召回候选中调整顺序，无法救回未召回目标。


