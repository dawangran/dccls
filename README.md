# dccls

`dccls` 是一个基于冻结 backbone + 分类头（MIL attention）的长序列分类训练脚本。

当前数据输入方式已经统一为：**按文件夹组织类别**。

---

## 1. 数据格式

### 1.1 目录结构

`--data_root` 目录下，每个一级子目录视为一个分类标签（class）。

```text
data_root/
  class_A/
    part-000.jsonl.gz
    part-001.jsonl.gz
  class_B/
    part-000.jsonl.gz
  class_C/
    xxx.jsonl
```

- 子目录名就是类别名。
- 每个子目录可包含一个或多个 `*.jsonl` / `*.jsonl.gz` 文件。

### 1.2 单条样本格式

每行一个 JSON 对象，至少包含：

- `id`: 样本唯一 ID
- `text`: 文本字段（支持字符串或 token 列表）

示例：

```json
{"id": "250F601969012_1_8048_1_3690_21421", "text": ["<|bwav:11797|>", "<|bwav:11097|>", "<|bwav:6750|>"]}
```

> `text` 如果是数组，会自动拼接为字符串再解析 `<|bwav:...|>` token。

---

## 2. 训练参数（关键）

核心变更：

- 不再使用 `--data`
- 不再使用 `--num_classes`
- 不再依赖 JSON 内的 `gene_id`
- 使用 `--reads_per_class` 控制“每个分类最多抽取多少条”

常用参数：

- `--data_root`：数据根目录（必填）
- `--model_path`：HF 模型路径（必填）
- `--outdir`：输出目录（必填）
- `--reads_per_class`：每类采样条数（默认 100）
- `--text_field`：文本字段名（默认 `text`）

---

## 3. 运行示例

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --reads_per_class 100 \
  --batch_size 8 \
  --epochs 30
```

---

## 4. 输出文件

训练前会写出类别与划分信息：

- `class2id.json`
- `id2class.json`
- `selected_class_counts.json`
- `split_map_7_2_1_reads{N}_salt{S}.json`
- `sampled_rids_by_class_reads{N}_salt{S}.json`

其中 `N` 为 `--reads_per_class`，`S` 为 `--split_salt`。

---

## 5. 依赖说明

至少需要：

- `python>=3.10`
- `torch`
- `numpy`
- `transformers`

可选：

- `wandb`（当 `--wandb` 时）

---

## 6. 常见问题

### Q1: 没有 `gene_id` 能训练吗？
可以。当前版本完全使用“文件夹名作为标签”，不再读取 `gene_id`。

### Q2: 每个类别样本不均衡怎么办？
可通过 `--reads_per_class` 统一每类最多采样数量；也可以启用 `--use_class_weight`。

### Q3: `text` 是数组而不是字符串怎么办？
已支持，脚本会自动兼容数组格式。
