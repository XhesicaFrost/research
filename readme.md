# 问答系统项目 README

## 项目概述

这是一个基于大语言模型的端到端问答系统项目，实现了从问题回答猜测、相关文档提取、文档摘要生成、答案预测到最终评估的完整流程。项目使用了多个先进的NLP模型，如Qwen3-0.6B、Llama-3.2-1B、BART等，提供高质量的自动化问答功能。

## 项目结构

```
research/
├── requirements.txt      # 项目依赖
├── guess.py             # 问题回答猜测
├── extract.py           # 相关文档提取
├── summary.py           # 文档摘要生成
├── answer.py            # 基于上下文的答案预测
├── judge.py             # 答案质量评估
├── data/                # 数据目录
└── outputs/             # 输出结果目录
```

## 环境配置

### 依赖安装

```bash
pip install -r requirements.txt
```

### 主要依赖库

- `torch>=1.9.0` - PyTorch深度学习框架
- `transformers>=4.21.0` - Hugging Face模型库
- `sentence-transformers` - 句子嵌入模型
- `tokenizers>=0.13.0` - 文本分词器
- `datasets>=2.0.0` - 数据集处理
- `numpy>=1.21.0` - 数值计算
- `pandas>=1.3.0` - 数据处理
- `tqdm>=4.62.0` - 进度条显示

## 文件功能详细说明

### 1. [`guess.py`](guess.py) - 问题回答猜测

**功能描述：**
- 基于问题生成初步的回答猜测
- 支持多种预训练模型：Qwen、Llama、GPT-2等
- 智能模型降级策略，确保系统稳定性

**主要特性：**
- 优先级模型加载（按性能排序）
- 智能答案提取和文本清理
- 完善的错误处理机制
- 支持GPU加速和CPU降级

**输入文件格式：**
```json
{"question": "问题内容", "answer": "标准答案", ...}
```

**输出文件格式：**
```json
{"question": "问题内容", "answer": "标准答案", "guess": "模型猜测答案", ...}
```

**更改输入文件路径：**
```python
# 在main()函数中修改
def main():
    input_file = "/path/to/your/question_file.json"  # 修改此路径
    output_dir = "/path/to/your/output/directory"    # 修改输出目录
    process_qa_file(input_file, output_dir, 100)    # 100为处理条数限制
```

### 2. [`extract.py`](extract.py) - 相关文档提取

**功能描述：**
- 使用语义相似度从大量文档中提取与问题最相关的段落
- 基于sentence-transformers模型进行向量化检索
- 支持批量处理和可配置的提取数量

**主要特性：**
- 高效的语义向量检索
- 可配置的相似度阈值和提取数量
- 进度条显示处理状态
- 支持大规模文档库检索

**输入文件格式：**
- 问答文件：`{"question": "问题", "guess": "猜测答案", ...}`
- 段落文件：`{"passage": "文档段落内容", ...}`

**输出文件格式：**
```json
{"question": "问题", "guess": "猜测答案", "related": ["相关段落1", "相关段落2", ...], ...}
```

**更改输入文件路径：**
```python
def main(
    passage_file="/path/to/your/passage_file.json",     # 文档段落文件
    qa_file="/path/to/your/qa_file.json",               # 问答文件
    output_dir="/path/to/your/output/directory",        # 输出目录
    passage_limit=600,                                  # 段落数量限制
    top_k=5                                            # 提取的相关段落数量
):
```

### 3. [`summary.py`](summary.py) - 文档摘要生成

**功能描述：**
- 将多个相关段落合并并生成简洁的摘要
- 支持多种摘要模型：BART、DistilBART、T5、Pegasus等
- 智能长度控制，避免超出模型限制

**主要特性：**
- 多模型支持和自动降级
- 智能文本长度管理
- 针对不同模型的优化处理
- 支持GPU/CPU自适应

**输入文件格式：**
```json
{"question": "问题", "related": ["相关段落1", "相关段落2", ...], ...}
```

**输出文件格式：**
```json
{"question": "问题", "related": ["段落1", "段落2", ...], "summary": "生成的摘要", ...}
```

**更改输入文件路径：**
```python
def main():
    input_file = "/path/to/your/extract_file.json"  # 修改输入文件路径
    process_extraction_file(input_file)
```

### 4. [`answer.py`](answer.py) - 基于上下文的答案预测

**功能描述：**
- 基于问题和摘要生成最终的预测答案
- 优先使用Qwen3-0.6B模型，支持多种备选模型
- 智能答案提取，处理模型特殊输出格式

**主要特性：**
- 先进的Qwen模型集成
- 智能提示词工程
- 特殊标记处理（如`<think>`标记）
- 多层次的答案质量控制
- 基于规则的备用答案提取

**输入文件格式：**
```json
{"question": "问题", "summary": "文档摘要", "answer": "标准答案", ...}
```

**输出文件格式：**
```json
{"question": "问题", "summary": "摘要", "answer": "标准答案", "pred_answer": "预测答案", ...}
```

**更改输入文件路径：**
```python
def main():
    input_file = "/path/to/your/summary_file.json"  # 修改输入文件路径
    process_summary_file(input_file)
```

### 5. [`judge.py`](judge.py) - 答案质量评估

**功能描述：**
- 全方位评估预测答案的质量
- 结合传统指标和AI模型的语义评分
- 提供详细的统计分析和性能报告

**主要特性：**
- 多维度评分系统：
  - 精确匹配（Exact Match）
  - 部分匹配（Partial Match）
  - 语义相似度（Semantic Similarity）
  - AI句子相似度评分
  - 自然语言推理评分
  - LLM语义评分
- 问题类型分类（时间、人物、地点、事实等）
- 答案质量等级评定
- 详细的统计报告和可视化分析

**输入文件格式：**
```json
{"question": "问题", "answer": "标准答案", "pred_answer": "预测答案", ...}
```

**输出文件格式：**
```json
{
  "question": "问题",
  "answer": "标准答案", 
  "pred_answer": "预测答案",
  "evaluation": {
    "exact_match_score": 0.0,
    "partial_match_score": 0.8,
    "semantic_similarity_score": 0.7,
    "ai_sentence_similarity_score": 0.85,
    "ai_nli_score": 0.9,
    "ai_llm_semantic_score": 0.88,
    "ai_semantic_average": 0.87,
    "composite_score": 0.84,
    "question_type": "person",
    "answer_quality": "good",
    "is_correct": true
  }
}
```

**更改输入文件路径：**
```python
def main():
    input_file = "/path/to/your/answer_file.json"  # 修改输入文件路径
    process_judge_file(input_file)
```
### 6. [`baseline.py`](answer.py) - 简单的对比示例

**功能描述：**
- 基于问题和关键词的回答
- 优先使用Qwen3-0.6B模型，支持多种备选模型
- 智能答案提取，处理模型特殊输出格式

**主要特性：**
- 先进的Qwen模型集成
- 智能提示词工程
- 特殊标记处理（如`<think>`标记）
- 多层次的答案质量控制
- 基于规则的备用答案提取

**输入文件格式：**
```json
{"question": "问题", "answer": "标准答案", ...}
```

**输出文件格式：**
```json
{"question": "问题", "answer": "标准答案", "pred_answer": "预测答案", ...}
```

**更改输入文件路径：**
```python
def main():
    input_file = "/path/to/your/summary_file.json"  # 修改输入文件路径
    process_summary_file(input_file)
```


## 完整工作流程

### 1. 数据准备
```bash
# 准备问答数据文件（JSON Lines格式）
# 准备文档段落文件（JSON Lines格式）
```

### 2. 运行完整流程
```bash
# 步骤1: 生成问题的初步回答
python guess.py

# 步骤2: 提取相关文档段落
python extract.py

# 步骤3: 生成文档摘要
python summary.py

# 步骤4: 基于摘要生成最终答案
python answer.py

# 步骤5: 评估答案质量
python judge.py
```

### 3. 输出文件命名规则
```
{日期}_answer_{输入文件名}      # guess.py输出
{日期}_extract_{输入文件名}     # extract.py输出
{日期}_summary_{输入文件名}     # summary.py输出
{日期}_answer_{输入文件名}      # answer.py输出
{日期}_judge_{输入文件名}       # judge.py输出
```

## 配置说明

### 模型配置

每个脚本都支持多种模型，按优先级尝试加载：

**guess.py模型优先级：**
1. Qwen/Qwen3-0.6B
2. microsoft/DialoGPT-medium
3. gpt2-medium
1. gpt2

**answer.py模型优先级：**
1. Qwen/Qwen3-0.6B
2. meta-llama/Llama-3.2-1B
3. distilbert-base-cased-distilled-squad
4. gpt2

**summary.py模型优先级：**
1. facebook/bart-large-cnn
2. sshleifer/distilbart-cnn-12-6
3. t5-small
4. google/pegasus-xsum

**baseline.py模型优先级：**
1. Qwen/Qwen3-0.6B
2. meta-llama/Llama-3.2-1B
3. distilbert-base-cased-distilled-squad
4. gpt2

### 硬件配置
- **GPU推荐：** 支持CUDA的GPU（8GB+ 显存）
- **CPU备选：** 所有模型都支持CPU运行
- **内存要求：** 16GB+ RAM推荐

## 性能优化建议

1. **GPU使用：** 确保安装了正确的CUDA版本
2. **批处理：** 对于大量数据，考虑分批处理
3. **模型选择：** 根据硬件配置选择合适的模型大小
4. **缓存优化：** 首次运行会下载模型，后续运行会使用本地缓存

## 故障排除

### 常见问题

1. **模型下载失败：**
   ```bash
   # 设置Hugging Face镜像
   export HF_ENDPOINT=https://hf-mirror.com
   ```

2. **GPU内存不足：**
   ```python
   # 在代码中强制使用CPU
   device = -1  # 而不是 device=0
   ```

3. **依赖版本冲突：**
   ```bash
   # 创建新的虚拟环境
   conda create -n qa_system python=3.8
   conda activate qa_system
   pip install -r requirements.txt
   ```

## 扩展开发

### 添加新模型
在相应的`load_*_model()`函数中添加新的模型名称到`models_to_try`列表。

### 自定义评估指标
在`judge.py`中的`calculate_scores()`函数中添加新的评分方法。

### 修改输出格式
在各个脚本的主处理函数中修改数据结构和保存逻辑。

## 许可证

本项目遵循MIT许可证。

## 贡献指南

欢迎提交Issue和Pull Request来改进本项目。

---

**注意：** 首次运行时会自动下载所需的预训练模型，请确保网络连接正常。模型文件较大，建议在网络条件良好的环境下运行。