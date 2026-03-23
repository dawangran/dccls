# dccls

`dccls` 是一个用于**长序列分类**的训练脚本：

- 底座模型使用 Hugging Face 模型作为 **frozen backbone**（训练时默认只前向，不更新 backbone 参数）。
- 每条 read / 样本会被切成多个 chunk。
- 每个 chunk 先经过 backbone 编码，再通过 **MIL attention 分类头** 聚合，输出最终类别。
- 当前版本的数据标签来源于：**`data_root` 下的一级子目录名**。

也就是说，这个项目适合这样的任务：

1. 每个样本是一条很长的 token 序列；
2. 每个类别下面有很多 `jsonl/jsonl.gz` 文件；
3. 你希望按类别目录自动采样、划分训练/验证/测试集并训练分类器。

---

## 1. 数据格式

### 1.1 目录结构

`--data_root` 目录下，**每个一级子目录就是一个类别**。

```text
data_root/
  class_A/
    part-000.jsonl.gz
    part-001.jsonl.gz
  class_B/
    part-000.jsonl.gz
  class_C/
    reads.jsonl
```

说明：

- `class_A` / `class_B` / `class_C` 这些目录名会直接作为类别名。
- 每个类别目录下可以放多个 `*.jsonl` 或 `*.jsonl.gz` 文件。
- 程序会自动扫描一级子目录，不需要额外提供标签映射文件。

---

### 1.2 单条样本格式

每行一个 JSON 对象，至少需要：

- `id`：样本唯一 ID，用于划分 train/val/test；
- `text`：文本字段，默认从这个字段里读 token 序列。

示例：

```json
{"id": "250F601969012_1_8048_1_3690_21421", "text": ["<|bwav:11797|>", "<|bwav:11097|>", "<|bwav:6750|>"]}
```

`text` 支持两种形式：

1. **字符串**

```json
{"id": "r1", "text": "<|bwav:10|> <|bwav:11|> <|bwav:12|>"}
```

2. **数组**（会先拼接成字符串）

```json
{"id": "r2", "text": ["<|bwav:10|>", "<|bwav:11|>", "<|bwav:12|>"]}
```

---

## 2. 整体流程

运行训练时，程序大致会做这些事：

1. 从 `data_root` 扫描所有类别目录和 `jsonl/jsonl.gz` 文件；
2. 用子目录名建立 `class2id`；
3. 对每个类别按 `--reads_per_class` 限制采样数量；
4. 按固定 7:2:1 比例划分 train / val / test；
5. 用 tokenizer 或 `<|bwav:...|>` 规则把 `text` 转成 token id；
6. 把一条长序列切成多个 chunk；
7. 每个 chunk 送入 HF backbone 得到 embedding；
8. 用 attention MIL 头聚合所有 chunk，输出类别；
9. 保存模型、划分文件、混淆矩阵、可选 attention 诊断结果。

---

## 3. 安装依赖

至少需要：

- `python>=3.10`
- `torch`
- `numpy`
- `transformers`

可选依赖：

- `wandb`：当使用 `--wandb` 时需要
- `matplotlib`：如果你希望保存混淆矩阵 / attention 图像，建议安装

一个简单示例：

```bash
pip install torch numpy transformers matplotlib wandb
```

---

## 4. 运行示例

### 4.1 最常用训练命令

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --reads_per_class 100 \
  --batch_size 8 \
  --epochs 30
```

### 4.2 只生成划分，不训练

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --write_split_map_only
```

这会生成类别映射和 train/val/test 划分文件，然后直接退出。

### 4.3 使用 gated attention 头

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output \
  --head_type gated \
  --gated_hidden 128 \
  --gated_attn_dropout 0.1 \
  --gated_temperature 1.0
```

---

## 5. 所有参数说明

下面按功能分类，把命令行参数逐个说明清楚。

> 入口文件是 `python -m dccls.main`。

### 5.1 数据输入相关

#### `--data_root`
- 类型：`str`
- 必填：**是**
- 作用：数据根目录。
- 程序会扫描这个目录下的一级子目录，并把**子目录名当成类别名**。
- 如果目录下面没有发现任何 `jsonl/jsonl.gz` 文件，程序会报错。

#### `--text_field`
- 类型：`str`
- 默认值：`text`
- 作用：指定每行 JSON 中，哪个字段存放样本文本/序列。
- 如果你的数据是：
  ```json
  {"id": "xx", "sequence": "<|bwav:1|> <|bwav:2|>"}
  ```
  那就需要传：
  ```bash
  --text_field sequence
  ```

#### `--reads_per_class`
- 类型：`int`
- 默认值：`100`
- 作用：每个类别**最多采样多少条 read** 参与本次实验。
- 用途：
  - 控制训练规模；
  - 限制大类样本过多；
  - 让不同类别尽量更平衡。
- 示例：
  - 如果某个类有 5000 条，传 `--reads_per_class 100`，只会从中选 100 条；
  - 如果某个类只有 60 条，则最多取 60 条。

#### `--split_salt`
- 类型：`str`
- 默认值：`0`
- 作用：控制采样和 train/val/test 划分的**确定性随机种子字符串**。
- 改这个值会改变：
  - 每类被选中的 read 集合；
  - train / val / test 的具体分配。
- 适合做多次不同切分实验。

#### `--write_split_map_only`
- 类型：flag
- 默认值：关闭
- 作用：只生成类别映射和数据划分文件，不进入训练。
- 适合先检查数据划分是否正确。

---

### 5.2 模型与 tokenizer 相关

#### `--model_path`
- 类型：`str`
- 必填：**是**
- 作用：Hugging Face 模型路径，可以是：
  - 本地目录；
  - 或 HF hub 可识别路径（前提是环境可访问）。
- 程序会用它同时加载：
  - `AutoTokenizer`
  - `AutoModel`

#### `--vocab_size`
- 类型：`int`
- 默认值：`None`
- 作用：手动指定词表大小。
- 当不传时，默认使用 tokenizer 的 `vocab_size`。
- 典型用途：
  - 你的 token id 范围比 tokenizer 原始词表更大；
  - 需要扩展 embedding 大小以覆盖更大的离散 token id。

#### `--pad_id`
- 类型：`int`
- 默认值：`None`
- 作用：padding token 的 id。
- 当不传时，会尝试使用 tokenizer 的 `pad_token_id`。
- 如果 tokenizer 没有 `pad_token_id`，你**必须手动传这个参数**，否则程序会报错。

#### `--hidden_layer`
- 类型：`int`
- 默认值：`-1`
- 作用：指定从 backbone 的哪一层 hidden states 取出来做 chunk pooling。
- 含义：
  - `-1`：最后一层；
  - `-2`：倒数第二层；
  - `-3`：倒数第三层；
  - 以此类推。
- 适合做不同层表示效果对比实验。

#### `--add_special_tokens` / `--no-add_special_tokens`
- 类型：布尔开关（BooleanOptionalAction）
- 默认值：`None`，程序内部会按 `False` 处理
- 作用：控制 tokenizer 编码时是否自动添加特殊 token。
- 常见理解：
  - 开启时，tokenizer 可能会加 BOS / EOS / CLS / SEP 等特殊符号；
  - 关闭时，只编码原始文本内容。
- 如果你的输入已经是手工构造好的 token 序列，通常建议保持默认（即不加）。

---

### 5.3 长序列切块相关

#### `--chunk_len`
- 类型：`int`
- 默认值：`64`
- 作用：每个 chunk 的长度，即每次送入 backbone 的 token 数。
- 增大后：
  - 单个 chunk 能覆盖更长上下文；
  - 但显存/算力开销也会提高。

#### `--stride`
- 类型：`int`
- 默认值：`48`
- 作用：滑窗切 chunk 时，相邻 chunk 起点之间的步长。
- 关系：
  - 如果 `stride < chunk_len`，chunk 之间会重叠；
  - 如果 `stride == chunk_len`，chunk 不重叠；
  - 如果 `stride > chunk_len`，会出现间隔采样。
- 当前默认 `48 < 64`，所以是**有重叠滑窗**。

#### `--K_chunks`
- 类型：`int`
- 默认值：`64`
- 作用：每条 read 最多保留多少个 chunk。
- 当一条 read 切出来的 chunk 数：
  - **少于 `K_chunks`**：会 pad 到固定长度；
  - **多于 `K_chunks`**：会按确定性规则均匀抽取 `K_chunks` 个。
- 这个参数直接影响：
  - 每条样本的表示容量；
  - 显存占用；
  - 注意力聚合头的输入规模。

---

### 5.4 训练过程相关

#### `--outdir`
- 类型：`str`
- 必填：**是**
- 作用：输出目录。
- 程序会在这里保存：
  - 类别映射；
  - 划分文件；
  - 训练日志相关结果；
  - 模型权重；
  - 混淆矩阵/attention 分析文件等。

#### `--epochs`
- 类型：`int`
- 默认值：`30`
- 作用：训练轮数。
- 一个 epoch 表示把训练集完整迭代一遍。

#### `--batch_size`
- 类型：`int`
- 默认值：`8`
- 作用：每个 batch 中包含多少条 read。
- 注意：这里不是 chunk 数，而是**read 数 / 样本数**。
- 每条 read 内部还会再展开成 `K_chunks × chunk_len` 的张量，所以实际显存占用会和 `chunk_len`、`K_chunks` 一起增长。

#### `--num_workers`
- 类型：`int`
- 默认值：`4`
- 作用：`DataLoader` 读取数据时使用的 worker 数。
- 提高它通常可以改善数据预处理吞吐，但也会增加 CPU / 内存开销。
- 如果在某些环境里多进程读数据不稳定，可以尝试设成 `0`。

#### `--lr`
- 类型：`float`
- 默认值：`2e-4`
- 作用：优化器学习率。
- 当前训练逻辑主要更新分类头，因此这个学习率主要影响分类头收敛速度。

#### `--weight_decay`
- 类型：`float`
- 默认值：`1e-2`
- 作用：权重衰减（L2 风格正则）。
- 用于抑制过拟合，让参数不要过大。

#### `--label_smoothing`
- 类型：`float`
- 默认值：`0.05`
- 作用：交叉熵中的标签平滑系数。
- 开启后不会把目标分布看成绝对 one-hot，有助于：
  - 缓解过拟合；
  - 提升模型校准性。
- 设成 `0` 就是普通交叉熵。

#### `--use_class_weight`
- 类型：flag
- 默认值：关闭
- 作用：根据训练集中每个类别的样本频次，自动计算类别权重并用于 loss。
- 适合类别不平衡场景。
- 当前权重趋势是：**样本越少的类，loss 权重越高**。

#### `--amp`
- 类型：flag
- 默认值：关闭
- 作用：开启自动混合精度训练（Automatic Mixed Precision）。
- 一般在 CUDA 环境下能：
  - 降低显存占用；
  - 提高训练吞吐。
- 如果你在 CPU 上跑，通常不会带来实际收益。

#### `--seed`
- 类型：`int`
- 默认值：`42`
- 作用：设置随机种子，尽量保证可复现。
- 会影响训练初始化、部分随机过程等。

---

### 5.5 分类头相关

#### `--head_type`
- 类型：`str`
- 默认值：`single`
- 可选值：`single`, `gated`
- 作用：选择 MIL attention 分类头类型。

两种模式：

1. `single`
   - 单门控 attention；
   - 参数更少；
   - 结构更简单。

2. `gated`
   - gated attention；
   - 表达能力更强；
   - 可配合下面几个 gated 参数一起调。

#### `--gated_hidden`
- 类型：`int`
- 默认值：`128`
- 作用：仅在 `--head_type gated` 时有意义。
- 表示 gated attention 中间隐藏层维度。
- 越大表示 attention 打分网络更强，但参数和计算也更多。

#### `--gated_attn_dropout`
- 类型：`float`
- 默认值：`0.1`
- 作用：仅在 `--head_type gated` 时有意义。
- 对 attention 权重做 dropout，帮助正则化。

#### `--gated_temperature`
- 类型：`float`
- 默认值：`1.0`
- 作用：仅在 `--head_type gated` 时有意义。
- 用于 softmax 前对 attention score 做温度缩放。
- 一般理解：
  - 温度更小：attention 更尖锐、更集中；
  - 温度更大：attention 更平滑、更分散。

---

### 5.6 attention 诊断与可解释性相关

#### `--save_attn`
- 类型：flag
- 默认值：关闭
- 作用：保存 attention 统计和样本结果。
- 会在验证/测试时额外输出 attention 分析文件，方便看模型更关注哪些 chunk。
- 开启后通常会带来少量额外开销。

#### `--attn_max_samples`
- 类型：`int`
- 默认值：`64`
- 作用：最多保存多少条样本的 attention 向量到文件。
- 用途：避免 attention 结果文件过大。

---

### 5.7 Weights & Biases 相关

#### `--wandb`
- 类型：flag
- 默认值：关闭
- 作用：开启 wandb 记录训练过程。

#### `--wandb_project`
- 类型：`str`
- 默认值：`nanopore-gene-class`
- 作用：wandb 的项目名。

#### `--wandb_name`
- 类型：`str`
- 默认值：空字符串
- 作用：当前实验的 run 名称。
- 不传时通常由 wandb 自动命名或保持默认行为。

#### `--wandb_tags`
- 类型：`str`
- 默认值：空字符串
- 作用：wandb 的标签字符串。
- 一般可以手动写成逗号分隔，便于实验筛选。

#### `--wandb_offline`
- 类型：flag
- 默认值：关闭
- 作用：让 wandb 以离线模式记录。
- 适合无外网环境，后续再同步。

---

## 6. 参数怎么搭配比较合适

### 6.1 小规模快速验证

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output_quick \
  --reads_per_class 20 \
  --K_chunks 16 \
  --chunk_len 64 \
  --batch_size 4 \
  --epochs 3
```

适合：
- 先检查流程能不能跑通；
- 看数据格式是否正确；
- 快速验证模型有没有学习信号。

### 6.2 类别不平衡时

```bash
python -m dccls.main \
  --data_root /path/to/data_root \
  --model_path /path/to/hf_model \
  --outdir /path/to/output_balance \
  --reads_per_class 200 \
  --use_class_weight
```

适合：
- 某些类别明显样本少；
- 你不希望模型过度偏向大类。

### 6.3 显存不够时

优先尝试调小这些参数：

- `--batch_size`
- `--K_chunks`
- `--chunk_len`

其次可以尝试：

- 开启 `--amp`

---

## 7. 输出文件说明

程序运行后，常见输出包括：

### 7.1 划分与类别映射

- `class2id.json`：类别名到类别 ID 的映射
- `id2class.json`：类别 ID 到类别名的映射
- `selected_class_counts.json`：每个类别统计到的样本数
- `split_map_7_2_1_reads{N}_salt{S}.json`：每个 read id 属于 train/val/test 哪个划分
- `sampled_rids_by_class_reads{N}_salt{S}.json`：每个类别最终被选中的 read id 列表

其中：

- `N` 对应 `--reads_per_class`
- `S` 对应 `--split_salt`

### 7.2 评估与诊断

根据训练和参数设置，可能还会看到：

- `val_confusion_matrix.npy`
- `val_per_class_acc.csv`
- `val_confusion_matrix.png`
- `test_confusion_matrix.npy`
- `test_per_class_acc.csv`
- `test_confusion_matrix.png`

如果开启 `--save_attn`，还可能包括：

- `val_attn_stats.npz`
- `val_attn_samples.jsonl`
- `val_attn_topm_mass.png`
- `val_attn_avg_by_pos.png`
- `test_attn_stats.npz`
- `test_attn_samples.jsonl`
- `test_attn_topm_mass.png`
- `test_attn_avg_by_pos.png`

---

## 8. 常见问题

### Q1: 没有 `gene_id` 能训练吗？
可以。

当前版本**完全不依赖 `gene_id`**，标签来自目录名，也就是 `data_root/类别名/*.jsonl` 这种组织方式。

### Q2: `text` 不是字符串，而是数组，能读吗？
可以。

当前实现支持 `text` 是数组，程序会先把数组元素拼成一个字符串再编码。

### Q3: 我的 JSON 里文本字段不叫 `text` 怎么办？
传 `--text_field 你的字段名` 即可。

### Q4: tokenizer 没有 `pad_token_id` 怎么办？
手动传 `--pad_id`。

例如：

```bash
--pad_id 0
```

### Q5: 类别特别不平衡怎么办？
优先尝试：

1. 用 `--reads_per_class` 限制每类最多采样数；
2. 再加 `--use_class_weight`；
3. 必要时做多组 `--split_salt` 对比实验。

### Q6: 为什么我改了 `--split_salt`，结果会变？
因为这个参数本来就是用来改变：

- 每类被抽中的样本；
- train/val/test 的划分。

它的目的就是让你可以做不同切分下的稳健性验证。

### Q7: `chunk_len`、`stride`、`K_chunks` 三个参数怎么理解？
可以把它们理解成：

- `chunk_len`：每个小片段有多长；
- `stride`：两个小片段之间隔多远开始切；
- `K_chunks`：每条 read 最多保留多少个小片段给分类头。

---

## 9. 推荐的排查顺序

如果训练跑不起来，建议按下面顺序检查：

1. 目录结构是否是 `data_root/class_x/*.jsonl`；
2. 每条 JSON 是否至少包含 `id` 和目标文本字段；
3. `--text_field` 是否写对；
4. `--model_path` 能否正确加载 tokenizer / model；
5. tokenizer 是否有 `pad_token_id`，没有就传 `--pad_id`；
6. 显存不够时先减小 `--batch_size` 和 `--K_chunks`；
7. 先用 `--write_split_map_only` 检查划分结果是否正常。

---

## 10. 一句话总结

如果你只想记住最核心的几个参数，可以记这几个：

- `--data_root`：数据根目录
- `--model_path`：HF 模型路径
- `--outdir`：输出目录
- `--reads_per_class`：每类最多采样多少条
- `--text_field`：JSON 里文本字段名
- `--chunk_len` / `--stride` / `--K_chunks`：长序列怎么切块
- `--head_type`：用哪种 attention 头
- `--use_class_weight`：类别不平衡时是否加权
- `--save_attn`：是否导出 attention 分析结果

