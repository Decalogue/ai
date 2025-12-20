# BookMemory 模型选型指南

基于 BookMemory 架构的主要任务，为每个任务推荐最适合的先进模型（2024-2025）。

---

## 模型概览（2025最新）

### 主流大模型对比

| 模型 | 上下文窗口 | 推理能力 | Agentic能力 | 成本（输入/输出） | 最佳适用场景 |
|------|-----------|---------|------------|-----------------|------------|
| **Gemini 3 Pro** | 1M tokens | ⭐⭐⭐⭐⭐ (91.9% GPQA) | ⭐⭐⭐⭐ | $2/$12 | 长文本处理、整本书分析 |
| **Gemini 3 Flash** | 1M tokens | ⭐⭐⭐⭐ | ⭐⭐⭐ | 更低 | 快速任务、批量处理 |
| **Claude Opus 4.5** | 200K tokens | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ (88.9% tool use) | $5/$25 | 精确推理、Agentic任务 |
| **Claude 3.5 Sonnet** | 200K tokens | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $3/$15 | 写作风格、对话生成 |
| **GPT-4o** | 128K tokens | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | $2.5/$10 | 通用任务、平衡性能 |
| **GPT-4 Turbo** | 128K tokens | ⭐⭐⭐⭐ | ⭐⭐⭐ | $10/$30 | 快速任务、成本优化 |

### 关键发现

1. **Gemini 3 Pro**：1M tokens 上下文是最大优势，适合整本书一次性处理
2. **Claude Opus 4.5**：Agentic 能力最强，适合需要精确推理和工具使用的任务
3. **Gemini 3 Flash**：速度快、成本低，适合快速批量处理

### 快速选择指南

**需要处理整本书（长上下文）？**
→ **Gemini 3 Pro**（1M tokens，成本低）

**需要精确推理和复杂分析？**
→ **Claude Opus 4.5**（Agentic能力强）

**需要快速批量处理？**
→ **Gemini 3 Flash**（速度快、成本低）

**需要高质量写作风格？**
→ **Claude 3.5 Sonnet**（写作自然）

**需要平衡性能和成本？**
→ **GPT-4o**（通用性强）

---

## 一、拆解层（Decomposition Layer）模型

### 1. 人物提取（Character Extraction）

#### 任务描述
- 命名实体识别（NER）：识别人物姓名
- 人物描述提取：提取人物外貌、性格等描述
- 人物属性推断：推断年龄、性别、角色等

#### 推荐模型

**首选：GPT-4o / Claude 3.5 Sonnet / Gemini 3 Pro** ⭐⭐⭐⭐⭐
- **理由**：
  - 强大的上下文理解能力
  - 可以同时完成NER、描述提取、属性推断
  - 支持中文人名识别（包括别名、昵称）
  - 可以理解复杂的人物描述
- **使用方式**：Prompt工程 + 结构化输出
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（适合整本书一次性处理）
  - 多语言理解强（91.8% MMLU）
  - 成本低（$2/$12 per million tokens）

**备选：Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 速度快，成本更低
  - 适合批量处理
- **适用场景**：需要快速处理大量书籍时

**备选：CRENER（中文NER专用）** ⭐⭐⭐⭐
- **理由**：
  - 专门针对中文NER优化
  - 字符关系增强，识别实体边界准确
  - 适合大规模批量处理
- **适用场景**：需要高精度中文人名识别时

**备选：spaCy + 微调BERT** ⭐⭐⭐
- **理由**：
  - 速度快，成本低
  - 可以针对书籍领域微调
- **适用场景**：批量处理、成本敏感

```python
# 推荐配置
character_extraction = {
    "primary": "gpt-4o",  # 或 "claude-3.5-sonnet" / "gemini-3-pro"
    "long_context": "gemini-3-pro",  # 1M tokens，适合整本书处理
    "fast": "gemini-3-flash",  # 快速批量处理
    "fallback": "gpt-4-turbo",
    "specialized": "CRENER"  # 中文NER专用
}
```

---

### 2. 人物关系识别（Relationship Extraction）

#### 任务描述
- 识别人物之间的关系类型（朋友/敌人/恋人/父子等）
- 计算关系强度
- 追踪关系发展时间线

#### 推荐模型

**首选：GLiREL（零样本关系抽取）** ⭐⭐⭐⭐⭐
- **理由**：
  - 零样本能力，无需训练即可识别新关系类型
  - 支持任意关系标签
  - 同时处理所有标签和实体对
  - 适合书籍中复杂的人物关系
- **论文**：GLiREL: A Generalist Model for Zero-Shot Relation Extraction

**备选：Claude Opus 4.5 / GPT-4o / Gemini 3 Pro** ⭐⭐⭐⭐⭐
- **理由**：
  - 强大的推理能力
  - 可以理解复杂的关系描述
  - 支持关系强度评估
- **Claude Opus 4.5 优势**：
  - Agentic 能力强（88.9% tool use）
  - 精确推理能力好
  - 适合复杂关系分析
- **Gemini 3 Pro 优势**：
  - 推理能力强（91.9% GPQA）
  - 成本低
- **使用方式**：Prompt工程 + JSON输出

**备选：Autoregressive Text-to-Graph Framework** ⭐⭐⭐⭐
- **理由**：
  - 联合实体和关系抽取
  - 生成图结构，适合构建关系网络
- **适用场景**：需要同时抽取实体和关系

```python
# 推荐配置
relationship_extraction = {
    "primary": "GLiREL",  # 零样本关系抽取
    "reasoning": "claude-opus-4.5",  # 复杂关系推理
    "cost_effective": "gemini-3-pro",  # 成本优化
    "fallback": "gpt-4o",
    "graph_based": "Autoregressive Text-to-Graph"
}
```

---

### 3. 情节线识别（Plotline Identification）

#### 任务描述
- 识别情节线（主线/支线/伏笔）
- 事件序列分析
- 因果关系识别
- 情节类型分类

#### 推荐模型

**首选：Gemini 3 Pro / GPT-4o / Claude Opus 4.5** ⭐⭐⭐⭐⭐
- **理由**：
  - 需要深度理解文本结构和逻辑
  - 可以识别复杂的情节发展
  - 支持因果关系推理
  - 可以理解伏笔和铺垫
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（可以处理整本书）
  - 推理能力强（91.9% GPQA）
  - 成本低（$2/$12 per million tokens）
- **Claude Opus 4.5 优势**：
  - 精确推理能力好
  - 适合复杂情节分析
- **使用方式**：分步骤Prompt（事件提取 → 聚类 → 关系识别）

**备选：GPT-4 Turbo / Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 成本更低
  - 性能足够
- **适用场景**：预算有限时

**备选：Claude 3 Opus** ⭐⭐⭐⭐
- **理由**：
  - 长文本理解能力强
  - 逻辑推理能力好
- **适用场景**：需要处理超长章节时（200K tokens）

```python
# 推荐配置
plotline_identification = {
    "primary": "gemini-3-pro",  # 1M tokens，整本书分析
    "reasoning": "claude-opus-4.5",  # 复杂推理
    "fallback": "gpt-4o",
    "cost_effective": "gemini-3-flash",  # 快速处理
    "long_context": "claude-3-opus"  # 200K tokens
}
```

---

### 4. 事件提取（Event Extraction）

#### 任务描述
- 提取关键事件
- 识别事件类型
- 提取事件参与者
- 识别事件重要性

#### 推荐模型

**首选：GPT-4o / Gemini 3 Pro** ⭐⭐⭐⭐⭐
- **理由**：
  - 可以理解事件的结构和重要性
  - 支持多事件提取
  - 可以识别事件之间的关联
- **Gemini 3 Pro 优势**：
  - 长上下文（1M tokens）
  - 成本低
- **使用方式**：结构化Prompt

**备选：Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 速度快
  - 成本低
- **适用场景**：批量事件提取

**备选：T5 / FLAN-T5** ⭐⭐⭐⭐
- **理由**：
  - 可以针对事件提取任务微调
  - 速度快，成本低
- **适用场景**：批量处理、需要微调时

```python
# 推荐配置
event_extraction = {
    "primary": "gpt-4o",
    "fallback": "flan-t5-large",  # 微调后使用
    "batch_processing": "t5-base"  # 批量处理
}
```

---

### 5. 主题提取（Theme Extraction）

#### 任务描述
- 提取书籍主题
- 主题分布分析
- 主题演化追踪
- 主题聚类

#### 推荐模型

**首选：Gemini 3 Pro / GPT-4o / Claude Opus 4.5** ⭐⭐⭐⭐⭐
- **理由**：
  - 可以理解抽象主题
  - 支持主题演化分析
  - 可以识别隐含主题
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（可以分析整本书的主题演化）
  - 多语言理解强
  - 成本低
- **Claude Opus 4.5 优势**：
  - 深度分析能力强
- **使用方式**：Prompt工程

**备选：主题模型（LDA/BERTopic）** ⭐⭐⭐
- **理由**：
  - 无监督方法
  - 可以自动发现主题
  - 成本低
- **适用场景**：初步主题发现

```python
# 推荐配置
theme_extraction = {
    "primary": "gpt-4o",
    "unsupervised": "BERTopic",  # 无监督主题发现
    "clustering": "LDA"  # 主题聚类
}
```

---

### 6. 章节识别与拆分（Chapter Segmentation）

#### 任务描述
- 识别章节边界
- 提取章节标题
- 章节内容提取

#### 推荐模型

**首选：Gemini 3 Pro / GPT-4o** ⭐⭐⭐⭐⭐
- **理由**：
  - 可以理解章节结构
  - 识别章节标题模式
  - 处理各种格式的章节标记
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（可以一次性处理整本书）
  - 成本低
- **使用方式**：Prompt + 正则表达式辅助

**备选：Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 速度快
  - 适合批量处理
- **适用场景**：批量处理大量书籍

**备选：规则+LLM混合** ⭐⭐⭐⭐
- **理由**：
  - 规则处理常见格式（速度快）
  - LLM处理复杂格式（准确）
- **适用场景**：批量处理大量书籍

```python
# 推荐配置
chapter_segmentation = {
    "primary": "gpt-4o",
    "hybrid": "rule_based + gpt-4o",  # 规则+LLM混合
    "batch": "rule_based"  # 批量处理用规则
}
```

---

## 二、检索层（Retrieval Layer）模型

### 7. 语义嵌入（Semantic Embedding）

#### 任务描述
- 生成人物、情节、主题的向量表示
- 支持语义相似度检索
- 跨书籍相似度计算

#### 推荐模型

**首选：text-embedding-3-large（OpenAI）** ⭐⭐⭐⭐⭐
- **理由**：
  - 性能优异（MTEB排行榜前列）
  - 支持多语言（包括中文）
  - 维度可调（256/512/1024/3072）
  - API稳定
- **维度推荐**：1024（平衡性能和成本）

**备选：BGE-large-zh-v1.5（中文优化）** ⭐⭐⭐⭐⭐
- **理由**：
  - 专门针对中文优化
  - 性能接近text-embedding-3-large
  - 开源免费
- **适用场景**：主要处理中文书籍

**备选：multilingual-e5-large** ⭐⭐⭐⭐
- **理由**：
  - 多语言支持好
  - 开源免费
  - 性能优秀
- **适用场景**：多语言书籍库

**备选：BGE-M3** ⭐⭐⭐⭐
- **理由**：
  - 支持多粒度（词/句/段落）
  - 多语言支持
  - 开源
- **适用场景**：需要多粒度检索时

```python
# 推荐配置
embedding_models = {
    "primary": "text-embedding-3-large",  # OpenAI
    "chinese_optimized": "BGE-large-zh-v1.5",
    "multilingual": "multilingual-e5-large",
    "multigranularity": "BGE-M3"
}
```

---

## 三、创作层（Creation Layer）模型

### 8. 大纲生成（Outline Generation）

#### 任务描述
- 生成书籍大纲
- 章节框架设计
- 人物关系规划
- 情节发展规划

#### 推荐模型

**首选：Claude Opus 4.5 / Gemini 3 Pro / GPT-4o** ⭐⭐⭐⭐⭐
- **理由**：
  - 需要结构化思维
  - 逻辑规划能力强
  - 可以理解参考书籍结构
  - 支持复杂大纲生成
- **Claude Opus 4.5 优势**：
  - Agentic 能力强，可以规划复杂结构
  - 精确推理
- **Gemini 3 Pro 优势**：
  - 推理能力强（91.9% GPQA）
  - 成本低
- **使用方式**：Few-shot Prompt + 结构化输出

**备选：GPT-4 Turbo / Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 成本更低
  - 性能足够
- **适用场景**：预算有限时

```python
# 推荐配置
outline_generation = {
    "primary": "gpt-4o",
    "fallback": "gpt-4-turbo",
    "cost_effective": "claude-3-haiku"  # 简单大纲
}
```

---

### 9. 章节生成（Chapter Generation）

#### 任务描述
- 生成完整章节内容
- 保持人物性格一致性
- 保持情节发展一致性
- 长文本生成（数千字）

#### 推荐模型

**首选：Gemini 3 Pro / GPT-4o / Claude Opus 4.5** ⭐⭐⭐⭐⭐
- **理由**：
  - 长文本生成能力强
  - 上下文理解好（可以记住前后章节）
  - 一致性保持好
  - 创意表达能力强
- **Gemini 3 Pro 优势**：
  - **1M tokens 上下文**（可以记住整本书的上下文）
  - 成本低（$2/$12 per million tokens）
  - 推理能力强
- **Claude Opus 4.5 优势**：
  - 200K tokens 上下文
  - 写作质量高
  - Agentic 能力强
- **使用方式**：多轮对话 + 上下文管理

**备选：Claude 3.5 Sonnet** ⭐⭐⭐⭐⭐
- **理由**：
  - 长文本能力优秀
  - 写作风格自然
  - 创意表达好
- **适用场景**：需要更好写作风格时

**备选：GPT-4 Turbo / Gemini 3 Flash** ⭐⭐⭐⭐
- **理由**：
  - 成本更低
  - 性能足够
- **适用场景**：预算有限时

```python
# 推荐配置
chapter_generation = {
    "primary": "gemini-3-pro",  # 1M tokens，记住整本书上下文
    "high_quality": "claude-opus-4.5",  # 200K tokens，高质量写作
    "style_focused": "claude-3.5-sonnet",
    "fallback": "gpt-4o",
    "cost_effective": "gemini-3-flash"  # 快速生成
}
```

---

### 10. 人物设定生成（Character Profile Generation）

#### 任务描述
- 生成人物属性
- 设计人物背景
- 规划人物弧线
- 生成人物描述

#### 推荐模型

**首选：GPT-4o / Claude 3.5 Sonnet** ⭐⭐⭐⭐⭐
- **理由**：
  - 可以生成丰富的人物设定
  - 理解人物关系需求
  - 设计合理的人物弧线
- **使用方式**：结构化Prompt

```python
# 推荐配置
character_generation = {
    "primary": "gpt-4o",
    "fallback": "claude-3.5-sonnet"
}
```

---

### 11. 一致性验证（Consistency Validation）

#### 任务描述
- 验证人物性格一致性
- 验证情节发展一致性
- 验证时间线一致性
- 验证主题一致性

#### 推荐模型

**首选：Claude Opus 4.5 / Gemini 3 Pro / GPT-4o** ⭐⭐⭐⭐⭐
- **理由**：
  - 逻辑分析能力强
  - 可以对比多个章节
  - 精确识别不一致之处
- **Claude Opus 4.5 优势**：
  - 精确推理能力好
  - 可以给出详细的不一致分析
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（可以对比整本书）
  - 推理能力强
  - 成本低
- **使用方式**：对比分析Prompt

**备选：Claude 3.5 Sonnet** ⭐⭐⭐⭐
- **理由**：
  - 分析能力强
  - 可以给出改进建议
- **适用场景**：需要详细分析报告时

```python
# 推荐配置
consistency_validation = {
    "primary": "gpt-4o",
    "detailed_analysis": "claude-3.5-sonnet"
}
```

---

## 四、分享层（Sharing Layer）模型

### 12. 播客脚本生成（Podcast Script Generation）

#### 任务描述
- 生成对话式播客脚本
- 关键情节讨论
- 人物分析
- 主题探讨

#### 推荐模型

**首选：Claude 3.5 Sonnet / Claude Opus 4.5** ⭐⭐⭐⭐⭐
- **理由**：
  - 对话风格自然
  - 可以生成多角色对话
  - 语气把握准确
- **Claude Opus 4.5 优势**：
  - Agentic 能力强，可以模拟多角色对话
  - 对话质量更高
- **使用方式**：Few-shot Prompt（示例对话）

**备选：GPT-4o / Gemini 3 Pro** ⭐⭐⭐⭐
- **理由**：
  - 性能优秀
  - 可以生成对话
- **适用场景**：需要更多创意时

```python
# 推荐配置
podcast_script = {
    "primary": "claude-3.5-sonnet",  # 对话风格更好
    "fallback": "gpt-4o"
}
```

---

### 13. 摘要生成（Summary Generation）

#### 任务描述
- 生成书籍摘要（短/中/长）
- 章节摘要
- 关键信息提取

#### 推荐模型

**首选：Gemini 3 Flash / GPT-4 Turbo** ⭐⭐⭐⭐⭐
- **理由**：
  - 摘要任务相对简单
  - 成本低
  - 性能足够
- **Gemini 3 Flash 优势**：
  - 速度快
  - 成本最低
- **使用方式**：Prompt工程

**备选：GPT-4o / Gemini 3 Pro** ⭐⭐⭐⭐
- **理由**：
  - 质量更高
  - 可以生成更详细的摘要
- **适用场景**：需要高质量摘要时

**备选：Claude 3 Haiku** ⭐⭐⭐⭐
- **理由**：
  - 速度快
  - 成本低
- **适用场景**：快速摘要生成

```python
# 推荐配置
summary_generation = {
    "primary": "gpt-4-turbo",  # 性价比高
    "high_quality": "gpt-4o",
    "fast": "claude-3-haiku"
}
```

---

### 14. 分析报告生成（Analysis Report Generation）

#### 任务描述
- 人物关系分析
- 情节结构分析
- 主题分析
- 写作风格分析

#### 推荐模型

**首选：Claude Opus 4.5 / Gemini 3 Pro / GPT-4o** ⭐⭐⭐⭐⭐
- **理由**：
  - 需要深度分析能力
  - 可以生成结构化报告
  - 支持多维度分析
- **Claude Opus 4.5 优势**：
  - Agentic 能力强，可以深度分析
  - 报告质量高
- **Gemini 3 Pro 优势**：
  - 1M tokens 上下文（可以分析整本书）
  - 推理能力强
  - 成本低
- **使用方式**：分步骤分析 + 报告生成

```python
# 推荐配置
analysis_report = {
    "primary": "gpt-4o",
    "fallback": "claude-3.5-sonnet"
}
```

---

## 五、综合推荐配置

### 生产环境推荐

```python
class BookMemoryModelConfig:
    """书籍记忆系统模型配置"""
    
    # 拆解层
    DECOMPOSITION = {
        "character_extraction": "gemini-3-pro",  # 1M tokens，整本书处理
        "relationship_extraction": "GLiREL",  # 或 "claude-opus-4.5"
        "plotline_identification": "gemini-3-pro",  # 1M tokens，整本书分析
        "event_extraction": "gemini-3-pro",
        "theme_extraction": "gemini-3-pro",  # 1M tokens，主题演化分析
        "chapter_segmentation": "gemini-3-pro"  # 1M tokens，整本书拆分
    }
    
    # 检索层
    RETRIEVAL = {
        "embedding": "text-embedding-3-large",  # 或 "BGE-large-zh-v1.5"
        "similarity_search": "cosine"  # 使用Qdrant
    }
    
    # 创作层
    CREATION = {
        "outline_generation": "claude-opus-4.5",  # Agentic 能力强
        "chapter_generation": "gemini-3-pro",  # 1M tokens，记住整本书上下文
        "character_generation": "gemini-3-pro",
        "consistency_validation": "claude-opus-4.5"  # 精确推理
    }
    
    # 分享层
    SHARING = {
        "podcast_script": "claude-opus-4.5",  # 对话质量高
        "summary": "gemini-3-flash",  # 快速、低成本
        "analysis_report": "gemini-3-pro"  # 1M tokens，整本书分析
    }
```

### 成本优化配置

```python
class CostOptimizedConfig:
    """成本优化配置"""
    
    DECOMPOSITION = {
        "character_extraction": "gpt-4-turbo",
        "relationship_extraction": "gpt-4-turbo",
        "plotline_identification": "gpt-4-turbo",
        "event_extraction": "t5-base",  # 微调后使用
        "theme_extraction": "BERTopic",  # 无监督
        "chapter_segmentation": "rule_based + gpt-4-turbo"
    }
    
    RETRIEVAL = {
        "embedding": "BGE-large-zh-v1.5"  # 开源免费
    }
    
    CREATION = {
        "outline_generation": "gpt-4-turbo",
        "chapter_generation": "gpt-4-turbo",
        "character_generation": "gpt-4-turbo",
        "consistency_validation": "gpt-4-turbo"
    }
    
    SHARING = {
        "podcast_script": "gpt-4-turbo",
        "summary": "claude-3-haiku",
        "analysis_report": "gpt-4-turbo"
    }
```

---

## 六、模型选择决策树

```
任务类型？
├─ 拆解任务
│  ├─ 人物提取 → GPT-4o（中文用CRENER）
│  ├─ 关系抽取 → GLiREL（零样本）或 GPT-4o
│  ├─ 情节识别 → GPT-4o
│  └─ 主题提取 → GPT-4o（或BERTopic无监督）
│
├─ 检索任务
│  └─ 语义嵌入 → text-embedding-3-large（中文用BGE-large-zh）
│
├─ 创作任务
│  ├─ 大纲生成 → GPT-4o
│  ├─ 章节生成 → GPT-4o（风格用Claude 3.5 Sonnet）
│  └─ 一致性验证 → GPT-4o
│
└─ 分享任务
   ├─ 播客脚本 → Claude 3.5 Sonnet（对话风格好）
   ├─ 摘要生成 → GPT-4 Turbo（性价比）
   └─ 分析报告 → GPT-4o
```

---

## 七、模型性能对比

| 任务 | 最佳模型 | 备选模型 | 成本 | 延迟 | 上下文 |
|------|---------|---------|------|------|--------|
| 人物提取 | Gemini 3 Pro | GPT-4o / CRENER | 低 | 中 | 1M |
| 关系抽取 | GLiREL | Claude Opus 4.5 | 低 | 低 | - |
| 情节识别 | Gemini 3 Pro | Claude Opus 4.5 | 低 | 中 | 1M |
| 事件提取 | Gemini 3 Pro | GPT-4o | 低 | 中 | 1M |
| 主题提取 | Gemini 3 Pro | GPT-4o | 低 | 中 | 1M |
| 语义嵌入 | text-embedding-3-large | BGE-large-zh | 低 | 低 | - |
| 大纲生成 | Claude Opus 4.5 | Gemini 3 Pro | 中 | 中 | 200K |
| 章节生成 | Gemini 3 Pro | Claude Opus 4.5 | 低 | 中 | 1M |
| 一致性验证 | Claude Opus 4.5 | Gemini 3 Pro | 中 | 中 | 200K/1M |
| 播客脚本 | Claude Opus 4.5 | Claude 3.5 Sonnet | 中 | 中 | 200K |
| 摘要生成 | Gemini 3 Flash | GPT-4 Turbo | 低 | 低 | - |
| 分析报告 | Gemini 3 Pro | Claude Opus 4.5 | 低 | 中 | 1M |

---

## 八、总结

### 核心推荐（2025更新）

1. **拆解层**：
   - **Gemini 3 Pro**（1M tokens，整本书处理，成本低）
   - GLiREL（关系抽取，零样本）
   - CRENER（中文NER专用）

2. **检索层**：
   - text-embedding-3-large（通用）
   - BGE-large-zh-v1.5（中文优化）

3. **创作层**：
   - **Gemini 3 Pro**（章节生成，1M tokens上下文）
   - **Claude Opus 4.5**（大纲生成、一致性验证，Agentic能力强）
   - Claude 3.5 Sonnet（风格优化）

4. **分享层**：
   - **Claude Opus 4.5**（播客脚本，对话质量高）
   - **Gemini 3 Flash**（摘要生成，快速低成本）
   - **Gemini 3 Pro**（分析报告，1M tokens整本书分析）

### 关键原则

- **长上下文任务**：优先使用 **Gemini 3 Pro**（1M tokens，成本低）
- **精确推理任务**：使用 **Claude Opus 4.5**（Agentic能力强）
- **快速任务**：使用 **Gemini 3 Flash**（速度快、成本低）
- **结构化任务**：使用专用模型（GLiREL、CRENER）
- **中文任务**：优先考虑中文优化模型（BGE、CRENER）

### 新模型优势总结

**Gemini 3 Pro**：
- ✅ 1M tokens 上下文（最大优势）
- ✅ 推理能力强（91.9% GPQA）
- ✅ 多语言理解好（91.8% MMLU）
- ✅ 成本低（$2/$12 per million tokens）
- ✅ 适合：整本书处理、长文本生成、大规模分析

**Gemini 3 Flash**：
- ✅ 速度快
- ✅ 成本更低
- ✅ 适合：快速任务、批量处理、摘要生成

**Claude Opus 4.5**：
- ✅ Agentic 能力强（88.9% tool use）
- ✅ 精确推理能力好
- ✅ 代码能力强（80.9% SWE-bench）
- ✅ 200K tokens 上下文
- ✅ 适合：复杂推理、大纲生成、一致性验证、高质量对话
