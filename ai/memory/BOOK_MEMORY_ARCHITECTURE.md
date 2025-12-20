# BookMemory：书籍专用记忆架构

## 一、设计理念

**"结构化拆解 + 关系图谱 + 时序记忆 + 创作支持"**

针对书籍领域的特殊需求，设计一个能够：
1. **拆解**：深度理解书籍的结构、人物、情节、主题
2. **记忆**：多维度、多层级地存储书籍知识
3. **分享**：支持多样化输出（播客、摘要、分析）
4. **创作**：辅助生成大纲、章节、人物设定

---

## 二、核心架构

```
┌─────────────────────────────────────────────────────────────┐
│                    BookMemory 书籍记忆系统                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  拆解层       │  │  存储层      │  │  关系层       │      │
│  │ (Decompose)  │  │  (Storage)   │  │  (Graph)     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│         │                 │                 │              │
│  书籍解析   │    FoA/DA/LTM  │   人物/情节/主题图谱          │
│  结构化提取 │    分层存储    │    动态关系网络               │
│         │                 │                 │              │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
    ┌─────┴─────┐    ┌─────┴─────┐    ┌─────┴─────┐
    │ 创作层     │    │ 检索层    │    │ 分享层     │
    │ (Create)  │    │ (Retrieve)│    │ (Share)   │
    └───────────┘    └───────────┘    └───────────┘
```

---

## 三、核心组件设计

### 1. 书籍结构化表示（Book Structure）

```python
@dataclass
class Book:
    """书籍整体表示"""
    id: str
    title: str
    author: str
    metadata: dict
        - genre: 类型（小说/非虚构/传记等）
        - theme: 主题
        - style: 风格
        - period: 时代背景
    chapters: List[Chapter]  # 章节列表
    characters: List[Character]  # 人物列表
    plotlines: List[Plotline]  # 情节线列表
    themes: List[Theme]  # 主题列表
    timeline: Timeline  # 时间线

@dataclass
class Chapter:
    """章节表示"""
    id: str
    book_id: str
    chapter_number: int
    title: str
    content: str
    summary: str  # 章节摘要
    key_events: List[Event]  # 关键事件
    characters_involved: List[str]  # 涉及的人物ID
    plotlines_involved: List[str]  # 涉及的情节线ID
    themes: List[str]  # 主题标签
    position_in_story: float  # 在故事中的位置（0.0-1.0）
    embedding: np.ndarray

@dataclass
class Character:
    """人物表示"""
    id: str
    book_id: str
    name: str
    aliases: List[str]  # 别名
    description: str  # 人物描述
    attributes: dict  # 属性
        - age: 年龄
        - gender: 性别
        - role: 角色（主角/配角/反派等）
        - personality: 性格特征
        - background: 背景
    relationships: List[Relationship]  # 人物关系
    appearances: List[str]  # 出现的章节ID
    character_arc: CharacterArc  # 人物弧线
    embedding: np.ndarray

@dataclass
class Relationship:
    """人物关系"""
    character1_id: str
    character2_id: str
    relation_type: str  # 关系类型（朋友/敌人/恋人/父子等）
    strength: float  # 关系强度（0.0-1.0）
    description: str  # 关系描述
    timeline: List[RelationshipEvent]  # 关系发展时间线

@dataclass
class Plotline:
    """情节线"""
    id: str
    book_id: str
    name: str
    description: str
    type: str  # 情节类型（主线/支线/伏笔等）
    chapters: List[str]  # 涉及的章节ID
    key_events: List[Event]  # 关键事件
    resolution: str  # 结局/解决方式
    embedding: np.ndarray

@dataclass
class Event:
    """事件"""
    id: str
    chapter_id: str
    description: str
    timestamp: float  # 在故事中的时间位置
    characters_involved: List[str]
    plotlines_involved: List[str]
    importance: float  # 重要性评分
    causal_links: List[str]  # 因果关联的事件ID

@dataclass
class Theme:
    """主题"""
    id: str
    book_id: str
    name: str
    description: str
    chapters: List[str]  # 相关章节
    characters: List[str]  # 相关人物
    embedding: np.ndarray
```

---

### 2. 拆解层（Decomposition Layer）

```python
class BookDecomposer:
    """
    书籍拆解器
    将原始书籍文本拆解为结构化表示
    """
    
    def decompose(self, book_text: str) -> Book:
        """
        拆解书籍
        1. 识别章节结构
        2. 提取人物信息
        3. 识别情节线
        4. 提取主题
        5. 构建时间线
        """
        # 1. 章节拆解
        chapters = self.extract_chapters(book_text)
        
        # 2. 人物提取
        characters = self.extract_characters(book_text, chapters)
        
        # 3. 情节识别
        plotlines = self.identify_plotlines(chapters)
        
        # 4. 主题提取
        themes = self.extract_themes(chapters)
        
        # 5. 事件提取
        events = self.extract_events(chapters)
        
        # 6. 关系构建
        relationships = self.build_relationships(characters, events)
        
        # 7. 时间线构建
        timeline = self.build_timeline(events, chapters)
        
        return Book(
            chapters=chapters,
            characters=characters,
            plotlines=plotlines,
            themes=themes,
            timeline=timeline
        )
    
    def extract_characters(self, text: str, chapters: List[Chapter]) -> List[Character]:
        """
        人物提取
        - 命名实体识别（NER）
        - 人物描述提取
        - 人物属性推断
        - 人物关系识别
        """
        pass
    
    def identify_plotlines(self, chapters: List[Chapter]) -> List[Plotline]:
        """
        情节线识别
        - 事件序列分析
        - 因果关系识别
        - 情节类型分类（主线/支线）
        """
        pass
    
    def extract_themes(self, chapters: List[Chapter]) -> List[Theme]:
        """
        主题提取
        - 主题词提取
        - 主题分布分析
        - 主题演化追踪
        """
        pass
```

---

### 3. 存储层（Storage Layer）- 基于 CogMem + LightMem

```python
class BookMemoryStorage:
    """
    书籍记忆存储系统
    分层存储：FoA（当前创作上下文）-> DA（相关书籍缓存）-> LTM（所有书籍库）
    """
    
    class FocusOfAttention:
        """
        注意力焦点（FoA）- 当前创作工作区
        - 容量：当前章节 + 相关人物 + 相关情节
        - 速度：极快（内存）
        - 内容：
          - 当前创作的章节
          - 当前章节涉及的人物
          - 当前章节涉及的情节线
          - 相关的前后章节
        """
        def __init__(self):
            self.current_chapter: Optional[Chapter] = None
            self.active_characters: List[Character] = []
            self.active_plotlines: List[Plotline] = []
            self.context_chapters: List[Chapter] = []  # 前后章节上下文
        
        def update_for_creation(self, chapter_context: dict):
            """更新为创作上下文"""
            # 加载当前章节
            # 加载相关人物
            # 加载相关情节
            # 加载前后章节
            pass
    
    class DirectAccess:
        """
        直接访问（DA）- 相关书籍快速缓存
        - 容量：最近访问的书籍 + 相关书籍
        - 速度：快（缓存）
        - 内容：
          - 最近拆解的书籍
          - 与当前创作相关的书籍
          - 相似类型的书籍
        """
        def __init__(self):
            self.recent_books: Dict[str, Book] = {}  # 最近访问
            self.related_books: Dict[str, Book] = {}  # 相关书籍
            self.similar_books: Dict[str, Book] = {}  # 相似书籍
        
        def cache_related(self, book: Book, similarity_threshold: float = 0.7):
            """缓存相关书籍"""
            pass
    
    class LongTermMemory:
        """
        长期记忆（LTM）- 所有书籍库
        - 容量：无限制
        - 速度：慢（数据库）
        - 结构：
          - 书籍库（PostgreSQL）
          - 向量库（人物、情节、主题的向量表示）
          - 图数据库（关系网络）
        """
        def __init__(self):
            self.book_db: Database  # 书籍数据库
            self.vector_db: VectorDB  # 向量数据库
            self.graph_db: GraphDB  # 图数据库
        
        def store_book(self, book: Book):
            """存储完整书籍"""
            # 1. 存储到关系数据库
            self.book_db.store(book)
            
            # 2. 存储向量表示
            self.vector_db.store_embeddings(book)
            
            # 3. 存储到图数据库
            self.graph_db.store_graph(book)
```

---

### 4. 关系层（Graph Layer）- 基于 A-Mem + Graph of Records

```python
class BookMemoryGraph:
    """
    书籍记忆图谱
    多维度关系网络：人物关系图、情节关系图、主题关系图
    """
    
    class CharacterGraph:
        """
        人物关系图
        - 节点：人物
        - 边：关系（类型、强度、时间）
        """
        def __init__(self):
            self.nodes: Dict[str, Character] = {}
            self.edges: List[Relationship] = []
        
        def add_character(self, character: Character):
            """添加人物节点"""
            pass
        
        def add_relationship(self, relationship: Relationship):
            """添加关系边"""
            pass
        
        def find_character_network(self, character_id: str, depth: int = 2):
            """查找人物关系网络（n度关系）"""
            pass
        
        def analyze_character_centrality(self):
            """分析人物中心性（重要性）"""
            pass
    
    class PlotlineGraph:
        """
        情节关系图
        - 节点：情节线、事件
        - 边：因果关系、时序关系、并行关系
        """
        def __init__(self):
            self.plotline_nodes: Dict[str, Plotline] = {}
            self.event_nodes: Dict[str, Event] = {}
            self.causal_edges: List[Tuple[str, str]] = []  # 因果关系
            self.temporal_edges: List[Tuple[str, str]] = []  # 时序关系
        
        def add_plotline(self, plotline: Plotline):
            """添加情节线"""
            pass
        
        def add_causal_link(self, cause_event_id: str, effect_event_id: str):
            """添加因果链接"""
            pass
        
        def find_plotline_path(self, start_event_id: str, end_event_id: str):
            """查找情节路径"""
            pass
    
    class ThemeGraph:
        """
        主题关系图
        - 节点：主题
        - 边：主题关联、主题演化
        """
        def __init__(self):
            self.theme_nodes: Dict[str, Theme] = {}
            self.association_edges: List[Tuple[str, str, float]] = []  # 主题关联
        
        def find_theme_clusters(self):
            """发现主题聚类"""
            pass
    
    class CrossBookGraph:
        """
        跨书籍关系图
        - 相似人物
        - 相似情节
        - 相似主题
        - 引用关系
        """
        def find_similar_characters(self, character: Character, top_k: int = 5):
            """查找相似人物（跨书籍）"""
            pass
        
        def find_similar_plotlines(self, plotline: Plotline, top_k: int = 5):
            """查找相似情节（跨书籍）"""
            pass
```

---

### 5. 检索层（Retrieval Layer）- 基于 Every Token Counts

```python
class BookMemoryRetrieval:
    """
    书籍记忆检索系统
    多维度、多层级检索
    """
    
    def retrieve_for_creation(self, query: dict) -> dict:
        """
        为创作检索相关记忆
        query: {
            'current_chapter': Chapter,
            'character_name': str,
            'plotline_name': str,
            'theme': str,
            'style': str
        }
        """
        results = {
            'characters': [],
            'plotlines': [],
            'themes': [],
            'similar_chapters': [],
            'reference_books': []
        }
        
        # 1. 从 FoA 检索（当前工作区）
        foa_results = self.foa.retrieve(query)
        results.update(foa_results)
        
        # 2. 从 DA 检索（相关书籍）
        if len(results['characters']) < 5:
            da_results = self.da.retrieve(query)
            results.update(da_results)
        
        # 3. 从 LTM 检索（所有书籍库）
        if len(results['characters']) < 10:
            ltm_results = self.ltm.retrieve(query)
            results.update(ltm_results)
        
        # 4. 通过关系图扩展
        graph_results = self.graph.expand(query)
        results.update(graph_results)
        
        return results
    
    def retrieve_by_character(self, character_name: str, book_id: str = None):
        """按人物检索"""
        # 1. 找到人物
        # 2. 找到相关章节
        # 3. 找到相关情节
        # 4. 找到人物关系网络
        pass
    
    def retrieve_by_plotline(self, plotline_name: str, book_id: str = None):
        """按情节线检索"""
        pass
    
    def retrieve_by_theme(self, theme: str, book_id: str = None):
        """按主题检索"""
        pass
    
    def retrieve_similar_books(self, book: Book, top_k: int = 5):
        """检索相似书籍"""
        # 基于：类型、主题、风格、人物、情节
        pass
```

---

### 6. 创作层（Creation Layer）

```python
class BookCreation:
    """
    书籍创作支持
    基于记忆系统生成大纲、章节、人物设定
    """
    
    def generate_outline(self, 
                        genre: str,
                        theme: str,
                        style: str,
                        reference_books: List[str] = None) -> Outline:
        """
        生成书籍大纲
        1. 检索相似书籍的结构
        2. 分析典型情节结构
        3. 生成章节框架
        4. 设计人物关系
        5. 规划情节发展
        """
        # 1. 检索参考书籍
        if reference_books:
            ref_books = [self.storage.ltm.get_book(bid) for bid in reference_books]
        else:
            ref_books = self.retrieval.retrieve_similar_books(
                Book(genre=genre, theme=theme, style=style)
            )
        
        # 2. 分析典型结构
        typical_structure = self.analyze_structure(ref_books)
        
        # 3. 生成大纲
        outline = self.create_outline(typical_structure, genre, theme)
        
        return outline
    
    def generate_chapter(self,
                        outline: Outline,
                        chapter_number: int,
                        context: dict) -> Chapter:
        """
        生成章节
        1. 检索相关记忆（人物、情节、前后章节）
        2. 保持一致性（人物性格、情节发展）
        3. 生成章节内容
        """
        # 1. 检索相关记忆
        memories = self.retrieval.retrieve_for_creation({
            'chapter_number': chapter_number,
            'outline': outline,
            **context
        })
        
        # 2. 更新 FoA（工作记忆）
        self.storage.foa.update_for_creation({
            'chapter_number': chapter_number,
            'characters': memories['characters'],
            'plotlines': memories['plotlines'],
            'previous_chapters': memories['similar_chapters']
        })
        
        # 3. 生成章节
        chapter = self.llm.generate_chapter(
            outline=outline,
            chapter_number=chapter_number,
            context=self.storage.foa.get_context(),
            memories=memories
        )
        
        # 4. 验证一致性
        self.validate_consistency(chapter, outline, memories)
        
        return chapter
    
    def generate_character_profile(self,
                                  role: str,
                                  relationships: List[str] = None,
                                  reference_characters: List[str] = None) -> Character:
        """
        生成人物设定
        1. 检索相似人物
        2. 分析人物关系需求
        3. 生成人物属性
        4. 设计人物弧线
        """
        pass
    
    def validate_consistency(self, chapter: Chapter, outline: Outline, memories: dict):
        """
        验证一致性
        - 人物性格一致性
        - 情节发展一致性
        - 时间线一致性
        - 主题一致性
        """
        pass
```

---

### 7. 分享层（Sharing Layer）

```python
class BookSharing:
    """
    书籍分享系统
    支持多种输出格式：播客、摘要、分析报告
    """
    
    def generate_podcast_script(self, book: Book, format: str = "conversational") -> str:
        """
        生成播客脚本
        - 对话式介绍
        - 关键情节讨论
        - 人物分析
        - 主题探讨
        """
        # 1. 提取关键信息
        key_points = self.extract_key_points(book)
        
        # 2. 生成对话脚本
        script = self.create_podcast_script(key_points, format)
        
        return script
    
    def generate_summary(self, book: Book, length: str = "medium") -> str:
        """
        生成摘要
        - 短摘要（1段）
        - 中摘要（1页）
        - 长摘要（详细）
        """
        pass
    
    def generate_analysis_report(self, book: Book) -> dict:
        """
        生成分析报告
        - 人物关系分析
        - 情节结构分析
        - 主题分析
        - 写作风格分析
        """
        report = {
            'character_analysis': self.analyze_characters(book),
            'plot_analysis': self.analyze_plot(book),
            'theme_analysis': self.analyze_themes(book),
            'style_analysis': self.analyze_style(book),
            'relationship_graph': self.graph.character_graph.visualize(),
            'plotline_timeline': self.graph.plotline_graph.get_timeline()
        }
        return report
```

---

## 四、完整工作流程

### 场景 1：书籍拆解与记忆

```
输入：原始书籍文本
    ↓
┌─────────────────┐
│ 1. 书籍拆解      │
│ - 章节识别       │
│ - 人物提取       │
│ - 情节识别       │
│ - 主题提取       │
└─────────────────┘
    ↓
┌─────────────────┐
│ 2. 结构化存储   │
│ - 存储到 LTM     │
│ - 生成向量表示   │
│ - 构建关系图     │
└─────────────────┘
    ↓
┌─────────────────┐
│ 3. 关系网络构建  │
│ - 人物关系图     │
│ - 情节关系图     │
│ - 主题关系图     │
└─────────────────┘
```

### 场景 2：创作支持

```
输入：创作需求（类型、主题、风格）
    ↓
┌─────────────────┐
│ 1. 检索参考      │
│ - 相似书籍       │
│ - 相似人物       │
│ - 相似情节       │
└─────────────────┘
    ↓
┌─────────────────┐
│ 2. 生成大纲      │
│ - 章节框架       │
│ - 人物设定       │
│ - 情节规划       │
└─────────────────┘
    ↓
┌─────────────────┐
│ 3. 逐章创作      │
│ - 检索相关记忆   │
│ - 更新 FoA       │
│ - 生成章节       │
│ - 验证一致性     │
└─────────────────┘
    ↓
┌─────────────────┐
│ 4. 记忆更新      │
│ - 存储新章节     │
│ - 更新关系图     │
│ - 更新时间线     │
└─────────────────┘
```

### 场景 3：内容分享

```
输入：书籍 + 分享类型（播客/摘要/分析）
    ↓
┌─────────────────┐
│ 1. 信息提取      │
│ - 关键情节       │
│ - 重要人物       │
│ - 核心主题       │
└─────────────────┘
    ↓
┌─────────────────┐
│ 2. 格式化输出    │
│ - 播客脚本       │
│ - 摘要文本       │
│ - 分析报告       │
└─────────────────┘
```

---

## 五、关键技术实现

### 1. 人物关系识别

```python
class CharacterRelationshipExtractor:
    """人物关系提取器"""
    
    def extract_relationships(self, text: str, characters: List[Character]) -> List[Relationship]:
        """
        提取人物关系
        1. 共现分析（出现在同一场景）
        2. 对话分析（对话频率、语气）
        3. 事件分析（共同参与的事件）
        4. 描述分析（相互描述）
        """
        relationships = []
        
        for char1, char2 in combinations(characters, 2):
            # 1. 共现频率
            cooccurrence = self.calculate_cooccurrence(char1, char2, text)
            
            # 2. 对话分析
            dialogue_analysis = self.analyze_dialogue(char1, char2, text)
            
            # 3. 关系类型推断
            relation_type = self.infer_relation_type(
                cooccurrence, dialogue_analysis
            )
            
            # 4. 关系强度计算
            strength = self.calculate_strength(
                cooccurrence, dialogue_analysis
            )
            
            relationships.append(Relationship(
                character1_id=char1.id,
                character2_id=char2.id,
                relation_type=relation_type,
                strength=strength
            ))
        
        return relationships
```

### 2. 情节线识别

```python
class PlotlineIdentifier:
    """情节线识别器"""
    
    def identify_plotlines(self, chapters: List[Chapter]) -> List[Plotline]:
        """
        识别情节线
        1. 事件序列分析
        2. 因果关系识别
        3. 主题一致性检查
        4. 人物参与分析
        """
        # 1. 提取所有事件
        all_events = []
        for chapter in chapters:
            all_events.extend(chapter.key_events)
        
        # 2. 事件聚类（基于主题、人物、时间）
        event_clusters = self.cluster_events(all_events)
        
        # 3. 构建情节线
        plotlines = []
        for cluster in event_clusters:
            plotline = Plotline(
                key_events=cluster,
                type=self.classify_plotline_type(cluster)
            )
            plotlines.append(plotline)
        
        return plotlines
```

### 3. 一致性验证

```python
class ConsistencyValidator:
    """一致性验证器"""
    
    def validate_character_consistency(self, 
                                     new_chapter: Chapter,
                                     existing_chapters: List[Chapter]) -> List[str]:
        """
        验证人物一致性
        - 性格一致性
        - 关系一致性
        - 背景一致性
        """
        issues = []
        
        for character in new_chapter.characters_involved:
            # 检查性格一致性
            if not self.check_personality_consistency(character, existing_chapters):
                issues.append(f"人物 {character.name} 性格不一致")
            
            # 检查关系一致性
            if not self.check_relationship_consistency(character, existing_chapters):
                issues.append(f"人物 {character.name} 关系不一致")
        
        return issues
    
    def validate_timeline_consistency(self, new_chapter: Chapter, timeline: Timeline):
        """验证时间线一致性"""
        pass
```

---

## 六、技术栈

### 存储
- **关系数据库**：PostgreSQL（书籍结构化数据）
- **向量数据库**：**Qdrant（推荐）** / Milvus（超大规模场景）
  - **推荐 Qdrant 的原因**：
    - 性能优异（Rust编写，低延迟、高QPS）
    - 高级过滤能力（按书籍/章节/人物/主题过滤）
    - 部署简单（单进程，无复杂依赖）
    - 资源效率（内存占用低）
    - 适合中大规模（<1亿向量，书籍记忆系统足够）
  - **Milvus 适用场景**：未来需要存储>1亿向量或需要企业级分布式架构
- **图数据库**：Neo4j（关系网络）
- **缓存**：Redis（FoA/DA）

### LLM 模型配置（多模型策略）

#### 默认模型配置

```python
class ModelConfig:
    """
    多模型配置策略
    不同任务使用最适合的模型，提升质量和效率
    """
    
    # 主模型（默认用于核心创作任务）
    DEFAULT_PRIMARY_MODEL = "gpt-4o"  # 或 "claude-3.5-sonnet"
    
    # 任务专用模型配置
    MODELS = {
        # 书籍拆解任务（需要深度理解）
        "decomposition": {
            "primary": "gpt-4o",  # 或 "claude-3.5-sonnet"
            "fallback": "gpt-4-turbo",
            "reason": "需要深度理解文本结构、人物关系、情节发展"
        },
        
        # 章节生成（核心创作任务）
        "chapter_generation": {
            "primary": "gpt-4o",  # 或 "claude-3.5-sonnet"
            "fallback": "gpt-4-turbo",
            "reason": "需要长文本生成、保持一致性、创意表达"
        },
        
        # 大纲生成（结构化任务）
        "outline_generation": {
            "primary": "gpt-4o",
            "fallback": "gpt-4-turbo",
            "reason": "需要结构化思维、逻辑规划"
        },
        
        # 人物设定生成
        "character_generation": {
            "primary": "gpt-4o",
            "fallback": "claude-3.5-sonnet",
            "reason": "需要人物塑造、性格一致性"
        },
        
        # 摘要生成（快速任务）
        "summary_generation": {
            "primary": "gpt-4-turbo",  # 或 "claude-3-haiku"
            "fallback": "gpt-3.5-turbo",
            "reason": "任务相对简单，使用快速模型降低成本"
        },
        
        # 播客脚本生成
        "podcast_script": {
            "primary": "claude-3.5-sonnet",  # 对话风格更好
            "fallback": "gpt-4o",
            "reason": "需要自然对话风格"
        },
        
        # 一致性验证（分析任务）
        "consistency_check": {
            "primary": "gpt-4o",
            "fallback": "claude-3.5-sonnet",
            "reason": "需要精确的逻辑分析"
        },
        
        # 关系抽取（结构化任务）
        "relation_extraction": {
            "primary": "gpt-4o",
            "fallback": "gpt-4-turbo",
            "reason": "需要精确的结构化输出"
        }
    }
    
    # 多模型集成策略（可选）
    ENSEMBLE_CONFIG = {
        "enabled": False,  # 默认关闭，需要高质量时开启
        "tasks": ["chapter_generation"],  # 仅对关键任务使用
        "models": ["gpt-4o", "claude-3.5-sonnet"],
        "strategy": "vote"  # 或 "weighted_average"
    }
```

#### 模型选择策略

**单模型 vs 多模型：**

1. **单模型模式（推荐用于开发/测试）**
   - 使用单一模型（如 GPT-4o）处理所有任务
   - 优点：简单、成本可控、一致性好
   - 缺点：某些任务可能不是最优选择
   - 适用：MVP阶段、预算有限

2. **多模型模式（推荐用于生产）**
   - 不同任务使用最适合的模型
   - 优点：质量最优、成本优化、任务适配
   - 缺点：配置复杂、需要管理多个API
   - 适用：生产环境、追求最佳效果

3. **集成模式（可选，用于关键任务）**
   - 多个模型生成结果，然后融合
   - 优点：质量最高、减少错误
   - 缺点：成本高、延迟增加
   - 适用：关键章节生成、重要决策

#### 推荐配置

**生产环境推荐：**
```python
# 核心创作任务：使用最强模型
chapter_generation: "gpt-4o" 或 "claude-3.5-sonnet"

# 快速任务：使用性价比模型
summary_generation: "gpt-4-turbo" 或 "claude-3-haiku"

# 对话任务：使用对话优化模型
podcast_script: "claude-3.5-sonnet"
```

**开发/测试推荐：**
```python
# 统一使用一个模型，简化配置
DEFAULT_MODEL = "gpt-4o"  # 或根据预算选择
```

### 专用模型
- **NER模型**：spaCy + 微调BERT（人物、地点、时间实体识别）
- **关系抽取模型**：微调BERT/LLaMA（人物关系、事件关系）
- **向量模型**：text-embedding-3-large 或 multilingual-e5-large（语义嵌入）

### 工具
- **文本处理**：spaCy, NLTK
- **图分析**：NetworkX, igraph
- **可视化**：D3.js, Cytoscape

---

## 七、优势总结

1. **深度理解**：结构化拆解，多维度分析
2. **关系网络**：人物、情节、主题的复杂关系建模
3. **创作支持**：基于记忆的智能创作辅助
4. **一致性保证**：多维度一致性验证
5. **多样化输出**：支持播客、摘要、分析等多种格式
6. **可扩展性**：模块化设计，易于扩展新功能

---

## 八、实现优先级

### Phase 1: 核心拆解与存储（MVP）
1. ✅ 书籍结构化拆解（章节、人物、情节）
2. ✅ 基础存储（LTM）
3. ✅ 简单关系图（人物关系）

### Phase 2: 检索与创作
4. ✅ 智能检索系统
5. ✅ 大纲生成
6. ✅ 章节生成
7. ✅ 一致性验证

### Phase 3: 高级功能
8. ✅ 复杂关系网络（情节关系、主题关系）
9. ✅ 跨书籍检索
10. ✅ 多样化分享（播客、分析报告）

---

## 九、示例用例

### 用例 1：拆解《三体》

```python
book = decomposer.decompose("三体.txt")

# 结果：
# - 3个主要人物：叶文洁、汪淼、史强
# - 多条情节线：三体入侵、科学探索、人性思考
# - 多个主题：科学、人性、文明、生存
# - 复杂人物关系网络
# - 完整时间线
```

### 用例 2：创作科幻小说

```python
# 1. 生成大纲
outline = creation.generate_outline(
    genre="科幻",
    theme="人工智能与人性",
    style="硬科幻",
    reference_books=["三体", "基地"]
)

# 2. 逐章创作
for i in range(1, 21):
    chapter = creation.generate_chapter(
        outline=outline,
        chapter_number=i,
        context={"previous_chapters": chapters[:i-1]}
    )
    chapters.append(chapter)
```

### 用例 3：生成播客脚本

```python
script = sharing.generate_podcast_script(
    book=book,
    format="conversational"
)

# 输出：对话式播客脚本，包含：
# - 书籍介绍
# - 关键情节讨论
# - 人物分析
# - 主题探讨
```
