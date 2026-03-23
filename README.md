# dccls

`dccls` 是一个用于**长序列分类**的训练脚本，整体结构为：**冻结 backbone + MIL attention 分类头**。

当前版本的数据组织方式已经统一为：**按文件夹名作为类别标签**。

---

## 1. 数据格式

### 1.1 目录结构

`--data_root` 下每个一级子目录都会被视为一个类别。

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

说明：

- 一级子目录名就是类别名。
- 每个类别目录下可放一个或多个 `*.jsonl` / `*.jsonl.gz` 文件。
- 脚本会自动递归发现这些文件，并用目录名生成 `class2id` / `id2class`。

### 1.2 单条样本格式

每行一个 JSON 对象，至少包含以下字段：

- `id`：样本唯一 ID。
- `text`：文本字段，支持：
  - 普通字符串。
  - token 列表，例如 `[..., ...]`。

示例：

```json
{"id": "250F601969012_1_8048_1_3690_21421", "text": ["<|bwav:11797|>", "<|bwav:11097|>", "<|bwav:6750|>"]}
```

补充说明：

- `text` 如果是数组，会先拼接成字符串，再继续解析。
- 默认读取字段为 `text`，可以通过 `--text_field` 改成别的字段名。
- 新增参数 `--min_text_length`，默认值为 **1000**。样本在编码后 token 数量小于该值时会被直接过滤，不参与类别统计、抽样、划分和训练。

---

## 2. 训练参数总览

和旧版本相比，当前脚本的关键变化如下：

- 不再使用 `--data`。
- 不再使用 `--num_classes`。
- 不再依赖 JSON 内部的 `gene_id`。
- 统一使用目录名作为标签。
- 使用 `--reads_per_class` 控制每个类别最多采样多少条数据。
- 新增 `--min_text_length` 控制最短文本长度过滤。
- 新增 `--warmup_ratio` 控制学习率 warmup 比例。
- 新增 `--unfreeze_last_n_layers`，用于选择“完全冻结基座”或“解冻最后 N 层”。
- 新增 `--supcon_weight` / `--supcon_temperature`，用于在 read embedding 上叠加 supervised contrastive loss。

下面是 `python -m dccls.main` 的主要参数说明。

### 2.1 数据与输入相关参数

#### `--data_root`
- 类型：`str`
- 必填：是
- 含义：数据根目录，每个一级子目录对应一个类别。

#### `--model_path`
- 类型：`str`
- 必填：是
- 含义：Hugging Face 模型路径，可以是本地路径或兼容路径。

#### `--outdir`
- 类型：`str`
- 必填：是
- 含义：训练输出目录，用于保存类别映射、划分文件、模型权重和评估结果。

#### `--reads_per_class`
- 类型：`int`
- 默认值：`100`
- 含义：每个类别最多采样多少条 read。
- 说明：如果某个类别样本数不足该值，则该类别会全部保留。

#### `--split_salt`
- 类型：`str`
- 默认值：`0`
- 含义：用于稳定抽样和划分的随机盐值。
- 说明：修改这个参数会改变每类样本的抽样结果以及 train/val/test 划分。

#### `--text_field`
- 类型：`str`
- 默认值：`text`
- 含义：样本中用于读取文本内容的字段名。

#### `--min_text_length`
- 类型：`int`
- 默认值：`1000`
- 含义：样本编码后 token 长度的最小值。
- 说明：
  - 小于该长度的样本会被过滤。
  - 过滤发生在类别统计、抽样划分和正式训练阶段。
  - 如果你希望保留更短样本，可以显式传入更小的值，例如 `--min_text_length 256`。

#### `--vocab_size`
- 类型：`int`
- 默认值：`None`
- 含义：词表大小。
- 说明：
  - 未指定时，默认读取 tokenizer 的 `vocab_size`。
  - 用于过滤非法 token id。

#### `--pad_id`
- 类型：`int`
- 默认值：`None`
- 含义：padding token id。
- 说明：
  - 未指定时，尝试从 tokenizer 的 `pad_token_id` 获取。
  - 如果 tokenizer 没有 pad token，则必须手动指定。

#### `--add_special_tokens` / `--no-add_special_tokens`
- 类型：布尔开关
- 默认值：`None`，内部最终默认按 `False` 处理
- 含义：调用 tokenizer 编码时，是否添加 special tokens。

---

### 2.2 序列切块相关参数

#### `--chunk_len`
- 类型：`int`
- 默认值：`64`
- 含义：每个 chunk 的长度。

#### `--stride`
- 类型：`int`
- 默认值：`48`
- 含义：滑窗切块时的步长。
- 说明：步长越小，chunk 重叠越多。

#### `--K_chunks`
- 类型：`int`
- 默认值：`64`
- 含义：每条 read 最多保留多少个 chunk。
- 说明：
  - 如果切出的 chunk 数超过该值，会做确定性采样。
  - 如果不足该值，会使用 padding 补齐并通过 mask 标识无效 chunk。

---

### 2.3 训练超参数

#### `--epochs`
- 类型：`int`
- 默认值：`30`
- 含义：训练轮数。

#### `--batch_size`
- 类型：`int`
- 默认值：`8`
- 含义：batch 大小，单位是 read 数，而不是 chunk 数。

#### `--num_workers`
- 类型：`int`
- 默认值：`4`
- 含义：DataLoader 的 worker 数量。

#### `--lr`
- 类型：`float`
- 默认值：`2e-4`
- 含义：AdamW 学习率。

#### `--weight_decay`
- 类型：`float`
- 默认值：`1e-2`
- 含义：AdamW 权重衰减。

#### `--warmup_ratio`
- 类型：`float`
- 默认值：`0.0`
- 含义：学习率 warmup 占总训练步数的比例。
- 说明：
  - 取值范围为 `[0, 1)`。
  - 例如设置为 `0.1` 表示前 10% 的训练 step 做线性 warmup，之后线性衰减。
  - 当前实现使用 `transformers.get_linear_schedule_with_warmup`。

#### `--label_smoothing`
- 类型：`float`
- 默认值：`0.05`
- 含义：交叉熵标签平滑系数。

#### `--use_class_weight`
- 类型：布尔开关
- 默认值：关闭
- 含义：是否根据训练集类别频次启用 class weight。
- 适用场景：类别不均衡时建议尝试打开。

#### `--supcon_weight`
- 类型：`float`
- 默认值：`0.0`
- 含义：read embedding 上 supervised contrastive loss 的权重。
- 说明：
  - 设为 `0.0` 时，行为与原始纯分类训练一致。
  - 可以从 `0.05`、`0.1`、`0.2` 开始尝试。
  - backbone 冻结时，SupCon 只会更新 MIL 头；如果解冻最后几层，SupCon 梯度也会传到这些已解冻的 backbone 参数。

#### `--supcon_temperature`
- 类型：`float`
- 默认值：`0.1`
- 含义：supervised contrastive loss 的温度系数。
- 说明：
  - 必须大于 `0`。
  - 常见起点可以是 `0.07` 或 `0.1`。

#### `--amp`
- 类型：布尔开关
- 默认值：关闭
- 含义：是否启用混合精度训练。
- 说明：通常在 CUDA 环境下更有意义。

#### `--seed`
- 类型：`int`
- 默认值：`42`
- 含义：随机种子。

---

### 2.4 Backbone / pooling / head 相关参数

#### `--hidden_layer`
- 类型：`int`
- 默认值：`-1`
- 含义：使用 backbone 的哪一层 hidden state 做 pooling。
- 说明：
  - `-1` 表示最后一层。
  - `-2` 表示倒数第二层。

#### `--unfreeze_last_n_layers`
- 类型：`int`
- 默认值：`0`
- 含义：控制 backbone 的训练方式。
- 说明：
  - `0`：完全冻结基座，只训练分类头。
  - `N > 0`：解冻 backbone 的最后 `N` 个 transformer block，并同时解冻输出投影层/归一化层。
  - 如果传入值大于 backbone 实际层数，则会自动退化为“解冻全部可识别的最后层”。
  - 实战建议先从 `0` 开始，只在验证集上限不足时再尝试 `1` 或 `2`。

#### `--head_type`
- 类型：`str`
- 默认值：`single`
- 可选值：`single`, `gated`
- 含义：MIL 分类头类型。

#### `--gated_hidden`
- 类型：`int`
- 默认值：`128`
- 含义：当 `--head_type gated` 时，gated attention 的隐藏层维度。

#### `--gated_attn_dropout`
- 类型：`float`
- 默认值：`0.1`
- 含义：gated attention 内部的 dropout 比例。

#### `--gated_temperature`
- 类型：`float`
- 默认值：`1.0`
- 含义：gated attention 的 softmax temperature。

---

### 2.5 Attention 分析输出相关参数

#### `--save_attn`
- 类型：布尔开关
- 默认值：关闭
- 含义：是否保存 attention 统计信息与样本。

#### `--attn_max_samples`
- 类型：`int`
- 默认值：`64`
- 含义：最多保存多少条 attention 样本用于分析。

---

### 2.6 运行模式与日志参数

#### `--write_split_map_only`
- 类型：布尔开关
- 默认值：关闭
- 含义：只写出抽样和划分文件，然后直接退出，不进入训练。

#### `--wandb`
- 类型：布尔开关
- 默认值：关闭
- 含义：是否启用 Weights & Biases 日志记录。

#### `--wandb_project`
- 类型：`str`
- 默认值：`nanopore-gene-class`
- 含义：wandb project 名称。

#### `--wandb_name`
- 类型：`str`
- 默认值：空字符串
- 含义：wandb run 名称。
- 说明：不传时会自动生成。

#### `--wandb_tags`
- 类型：`str`
- 默认值：空字符串
- 含义：wandb 标签，多个标签使用逗号分隔。

#### `--wandb_offline`
- 类型：布尔开关
- 默认值：关闭
- 含义：是否使用 wandb offline 模式。

---

## 3. 运行示例

### 3.1 最小训练示例

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output
```

### 3.2 带常用超参数的训练示例

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --reads_per_class 100 \
  --min_text_length 1000 \
  --chunk_len 64 \
  --stride 48 \
  --K_chunks 64 \
  --batch_size 8 \
  --epochs 30 \
  --lr 2e-4 \
  --warmup_ratio 0.1 \
  --unfreeze_last_n_layers 2 \
  --head_type gated \
  --supcon_weight 0.1 \
  --supcon_temperature 0.1
```

### 3.3 只生成划分文件

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --write_split_map_only
```

---

## 4. 输出文件

训练前或训练过程中通常会生成以下文件：

- `class2id.json`：类别名到类别 ID 的映射。
- `id2class.json`：类别 ID 到类别名的映射。
- `selected_class_counts.json`：过滤后每个类别可用样本数量统计。
- `split_map_7_2_1_reads{N}_salt{S}.json`：样本划分结果。
- `sampled_rids_by_class_reads{N}_salt{S}.json`：每个类别被采样的 read ID 列表。
- `best.pt`：验证集表现最优的分类头权重。
- `val_eXXX_confusion_matrix.npy/csv/png`：验证集混淆矩阵及各类准确率。
- `test_best_confusion_matrix.npy/csv/png`：测试集结果。
- 启用 `--save_attn` 时还会写出 attention 统计与可视化文件。

其中：

- `N` 对应 `--reads_per_class`。
- `S` 对应 `--split_salt`。

---

## 5. 依赖说明

至少需要：

- `python>=3.10`
- `torch`
- `numpy`
- `transformers`

可选依赖：

- `wandb`：当启用 `--wandb` 时需要。
- `matplotlib`：当保存混淆矩阵图或 attention 图时建议安装。

---

## 6. 常见问题

### Q1：没有 `gene_id` 还能训练吗？
可以。当前版本完全使用**文件夹名作为标签**，不再依赖 `gene_id`。

### Q2：每个类别样本不均衡怎么办？
可以先用 `--reads_per_class` 限制每类最多采样数量；如果仍然不均衡，可以再打开 `--use_class_weight`。

### Q3：`text` 是数组而不是字符串怎么办？
已经支持。脚本会先把数组拼接成字符串，再做后续编码处理。

### Q4：为什么有些样本没有进入训练？
常见原因有：

- 缺少 `id`。
- `text_field` 指向的字段为空。
- 编码后的 token 长度小于 `--min_text_length`。
- 文件不在有效类别目录下。

### Q5：`warmup_ratio` 应该怎么设？
一般可以从 `0.03`、`0.05` 或 `0.1` 开始尝试；如果训练非常短，可以先保持默认 `0.0`。

### Q6：怎么选择“冻结基座”还是“解冻最后几层”？
- 如果想保持原来行为，直接使用默认值 `--unfreeze_last_n_layers 0`。
- 如果想做轻量微调，可以尝试 `--unfreeze_last_n_layers 1`、`2` 或 `4`。
- 一般来说，解冻层数越多，显存占用和训练不稳定性也会更高。
- 更推荐的顺序是：先跑“冻结 backbone + MIL(+SupCon)”基线，再看是否需要解冻最后 `1~2` 层抬高上限。

### Q7：什么时候建议打开 supervised contrastive loss？
- 如果你希望 read embedding 的类内更紧、类间更分开，可以尝试打开 `--supcon_weight`。
- 推荐先保留 MIL 头，再用较小权重（如 `0.05` 或 `0.1`）叠加。
- 如果 batch 太小且同类样本在同一 batch 中很少，SupCon 的收益可能会不稳定。
