# Agent & Memory

| 论文 | 内容 | URL |
|:----:|:----|:----|
| Adaptation-of-Agentic-AI | 将提示工程（Prompt Engineering） 和微调（Fine-Tuning） 统一归纳为“适配”的两种形式。如果说基础模型是通用的“大脑”，适配就是让它能够精准指挥工具这一“肢体”的神经系统。 | https://arxiv.org/pdf/2510.00615 |
| HindSight | 一、记忆结构化：不再把记忆仅仅当作外挂硬盘，而是将其构建为一个结构化的、支持推理的一等公民。<br>二、三大核心操作：Retain（保留）：将对话流转化为结构化的记忆；Recall（回忆）：根据当前需求检索相关记忆；Reflect（反思）：基于记忆进行推理，回答问题，并更新信念。<br>三、TEMPR：构建时空感知的记忆图谱 CARA：带有个性的推理引擎 | http://arxiv.org/abs/2512.12818v1 |
| CogMem | 不再依赖单纯的上下文堆砌，而是模仿人类的认知机制，通过构建分层的、持久化的记忆系统，让大模型在长对话中也能保持清醒的头脑和连贯的逻辑。<br>一、记忆三重奏：1. 注意力焦点（Focus of Attention, FoA）；2. 直接访问记忆（Direct Access, DA）；3. 长期记忆（Long-Term Memory, LTM）<br>二、双智能体协作：推理与记忆分离 | http://arxiv.org/abs/2512.14118v1 |
| ACON | ACON是一个统一的框架，专门用于优化长时域（long-horizon）大型语言模型（LLM）代理（agents）的上下文压缩。它能够同时压缩环境观察（observations）和交互历史（interaction histories）。<br>ACON提出将优化后的压缩器（compressor）蒸馏到更小的模型中，从而显著降低计算开销。实验表明，经过蒸馏的小模型在压缩性能上能够保留超过95%的教师模型（teacher model）性能，同时减少了模块的额外开销。 | https://arxiv.org/pdf/2510.00615 |
| Titans | 一个具体可用的AI架构，首次实现了"测试时训练"的长期记忆系统。<br>短期工作记忆保证精准，长期记忆提供容量，固有知识提供稳定性。 | https://arxiv.org/pdf/2501.00663 |
| MIRAS | 将Titans的成功升华为通用理论,统一了所有序列模型的本质 | https://arxiv.org/abs/2504.13173 |
| Memory in the Age of AI Agents: A Survey | 对当前AI智能体（AI Agents）记忆机制最系统、最前沿的梳理 | https://arxiv.org/pdf/2512.13564 |
| Graph of Records | 利用图结构来管理 RAG 过程中产生的 LLMMemory，以增强长上下文理解效果。 | https://arxiv.org/pdf/2410.11001v2 |
| Every Token Counts (HSA-UltraLong) | 通过端到端可学习的检索机制+长度泛化设计，让AI拥有真正的长期记忆。 | https://arxiv.org/pdf/2511.23319 |
| LightMem | LightMem 通过模仿人类记忆的三层结构，打造了三个可以互相配合的轻量模块。<br>第一个轻量模块是感觉记忆过滤器（Light1）。<br>第二个轻量模块是短时记忆话题管家（Light2）。<br>第三个轻量模块是长时记忆与睡眠时间更新器，这也是 LightMem 最巧妙的创新。 | https://arxiv.org/pdf/2510.18866 |
| Evo-Memory | 一个用于评估大型语言模型（LLM）在测试时学习与自进化记忆（self-evolving memory）能力的全新基准与框架。<br>支持持续经验复用、动态记忆演化与任务级自我改进。 | https://arxiv.org/pdf/2511.20857 |
| MemMachine | MemMachine是一个开源的AI Agent通用记忆层，能让AI应用学习、存储和回忆过往会话的数据和偏好，提升交互体验。它支持短期、长期和个性化多种记忆类型。<br>MemMachine是由多个专门组件构成：情景记忆负责存储用户和Agent的具体交互事件；语义记忆通过层级结构归纳用户长期信息；用户画像记忆形成关于用户的持续画像；多模型访问层（MCP / RESTful / SDK）让它能被任何Agent系统调用。 | https://github.com/MemMachine/MemMachine |
| HaluMem | 业内首个操作级幻觉评测基准——HaluMem，从评测粒度、任务深度到数据规模全面突破 | https://arxiv.org/pdf/2511.03506 |
| A-Mem | 一种新颖的代理记忆系统（A-MEM），旨在为大型语言模型（LLM）代理提供动态组织和进化记忆的能力。<br>Zettelkasten方法：受Zettelkasten方法的启发，A-MEM通过动态索引和链接机制创建互联的知识网络。每个新记忆被构建成一个包含多个结构化属性的综合笔记，包括上下文描述、关键词和标签。<br>记忆生成：当添加新记忆时，系统会生成一个全面的笔记，并通过动态索引和链接建立与其他记忆的联系。新记忆的加入可以触发对现有历史记忆的上下文表示和属性的更新。<br>记忆进化：A-MEM通过分析历史记忆库来识别相关连接，并在新的记忆被整合时自动更新现有记忆的上下文表示和属性，从而实现记忆网络的持续优化。 | https://arxiv.org/pdf/2502.12110 |
| Chain-of-Agents | 单模型模拟多智能体协作！OPPO提出Chain-of-Agents范式 | https://arxiv.org/pdf/2508.13167 |
| Context Engineering: Sessions & Memory | Google最新发布的技术白皮书《Context Engineering: Sessions, Memory》揭示了一种突破性的解决方案——上下文工程，它正在重新定义如何构建具有长期记忆的智能代理。 | https://www.kaggle.com/whitepaper-context-engineering-sessions-and-memory |
| EverMemOS | EverMind 团队近日宣布正式发布其旗舰产品 EverMemOS，这是一款面向人工智能智能体的世界级长期记忆操作系统。<br>它旨在成为未来智能体的数据基础设施，为 AI 赋予持久、连贯、可进化的 “灵魂”。 | https://github.com/EverMind-AI/EverMemOS |
| O-Mem | OPPO AI Agent Team 提出O-Mem—— 一种基于主动用户画像的新型记忆框架，旨在解决 LLM 驱动智能体在复杂环境中长期交互的上下文一致性与动态个性化难题。<br>它通过从用户主动交互中动态提取和更新用户特征与事件记录，支持人物属性和主题相关上下文的分层检索。 | https://arxiv.org/pdf/2511.13593 |
| STMA | STMA（Spatio-Temporal Memory Agent）是一种为长程具身任务规划设计的新型框架，旨在通过整合时空记忆来提升智能体在动态环境中的任务规划与执行能力。 | https://arxiv.org/pdf/2502.10177 |
