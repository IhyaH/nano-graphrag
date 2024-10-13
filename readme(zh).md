😭 [GraphRAG](https://arxiv.org/pdf/2404.16130) 很好且功能强大，但官方的 [实现](https://github.com/microsoft/graphrag/tree/main) 阅读或修改起来很困难/痛苦。

😊 这个项目提供了一个 **更小、更快、更干净的 GraphRAG**，同时保留了核心功能（见 [基准测试](#benchmark) 和 [问题](#Issues)）。

🎁 不包括 `tests` 和提示，`nano-graphrag` 大约有 **1100行代码**。

👌 小巧但 [**便携**](#Components)（faiss, neo4j, ollama...），[**异步**](#Async) 且完全类型化。

## 安装

**从源代码安装**（推荐）

```shell
# 首先克隆这个仓库
cd nano-graphrag
conda install -c conda-forge hnswlib
pip install -e .
```

**从PyPi安装**

```shell
pip install nano-graphrag
```

## 快速开始

> [!TIP]
>
> **请在环境变量中设置OpenAI API密钥：`export OPENAI_API_KEY="sk-..."`。**

> [!TIP]
> 如果你使用的是Azure OpenAI API，请参考 [.env.example](./.env.example.azure) 设置你的azure openai。然后传递 `GraphRAG(...,using_azure_openai=True,...)` 以启用。

> [!TIP]
> 如果你没有密钥，可以查看这个 [示例](./examples/no_openai_key_at_all.py)，它使用了 `transformers` 和 `ollama`。如果你想使用另一个LLM或嵌入模型，请查看 [高级](#Advances)。

下载查尔斯·狄更斯的《圣诞颂歌》的副本：

```shell
curl https://raw.githubusercontent.com/gusye1234/nano-graphrag/main/tests/mock_data.txt  > ./book.txt
```

使用下面的Python代码片段：

```python
from nano_graphrag import GraphRAG, QueryParam

graph_func = GraphRAG(working_dir="./dickens")

with open("./book.txt") as f:
    graph_func.insert(f.read())

# 执行全局graphrag搜索
print(graph_func.query("这个故事中的主题是什么？"))

# 执行局部graphrag搜索（我认为这是更好、更可扩展的一个）
print(graph_func.query("这个故事中的主题是什么？", param=QueryParam(mode="local")))
```

下次你从同一个 `working_dir` 初始化一个 `GraphRAG` 时，它会自动重新加载所有的上下文。

#### 批量插入

```python
graph_func.insert(["TEXT1", "TEXT2",...])
```

<details>
<summary>增量插入</summary>

`nano-graphrag` 支持增量插入，不会有重复的计算或数据被添加：

```python
with open("./book.txt") as f:
    book = f.read()
    half_len = len(book) // 2
    graph_func.insert(book[:half_len])
    graph_func.insert(book[half_len:])
```

> `nano-graphrag` 使用内容的md5哈希作为键，所以不会有重复的块。
>
> 但是，每次你插入时，图的社区会被重新计算，社区报告也会被重新生成。

</details>

<details>
<summary>朴素RAG</summary>

`nano-graphrag` 也支持朴素RAG的插入和查询：

```python
graph_func = GraphRAG(working_dir="./dickens", enable_naive_rag=True)
...
# 查询
print(rag.query(
      "这个故事中的主题是什么？",
      param=QueryParam(mode="naive")
)
```

</details>

### 异步

对于每个方法 `NAME(...)`，都有一个对应的异步方法 `aNAME(...)`

```python
await graph_func.ainsert(...)
await graph_func.aquery(...)
...
```

### 可用参数

`GraphRAG` 和 `QueryParam` 是Python中的 `dataclass`。使用 `help(GraphRAG)` 和 `help(QueryParam)` 查看所有可用参数！或者查看 [高级](#Advances) 部分以了解一些选项。

## 组件

以下是你可以使用的组件：

| 类型            |                             是什么                             |                       在哪里                      |
| :-------------- | :----------------------------------------------------------: | :-----------------------------------------------: |
| LLM             |                            OpenAI                            |                    内置                     |
|                 |                           DeepSeek                           |              [示例](./examples)              |
|                 |                           `ollama`                           |              [示例](./examples)              |
| 嵌入            |                            OpenAI                            |                    内置                     |
|                 |                    Sentence-transformers                     |              [示例](./examples)              |
| 向量数据库       | [`nano-vectordb`](https://github.com/gusye1234/nano-vectordb)  |                    内置                     |
|                 |        [`hnswlib`](https://github.com/nmslib/hnswlib)         |         内置，[示例](./examples)         |
|                 |  [`milvus-lite`](https://github.com/milvus-io/milvus-lite)    |              [示例](./examples)              |
|                 | [faiss](https://github.com/facebookresearch/faiss?tab=readme-ov-file)  |              [示例](./examples)              |
| 图存储          | [`networkx`](https://networkx.org/documentation/stable/index.html)  |                    内置                     |
|                 |                [`neo4j`](https://neo4j.com/)                  | 内置([文档](./docs/use_neo4j_for_graphrag.md)) |
| 可视化          |                           graphml                            |              [示例](./examples)              |
| 分块            |                       按令牌大小                         |                    内置                     |
|                 |                      按文本分割器                       |                    内置                      |

- `内置` 意味着我们在 `nano-graphrag` 内部有该实现。`示例` 意味着我们在 [示例](./examples) 文件夹下的教程中有该实现。

- 查看 [examples/benchmarks](./examples/benchmarks) 以了解组件之间的一些比较。
- **总是欢迎贡献更多组件。**

## 高级

<details>
<summary>一些设置选项</summary>

- `GraphRAG(...,always_create_working_dir=False,...)` 将跳过目录创建步骤。如果你将所有组件切换到非文件存储，可以使用它。

</details>

<details>
<summary>只查询相关上下文</summary>

`graph_func.query` 返回最终答案而不进行流式传输。

如果你想在项目中与 `nano-graphrag` 交互，你可以使用 `param=QueryParam(..., only_need_context=True,...)`，它只会返回从图中检索到的上下文，类似于：

````
# 本地模式
-----报告-----
```csv
id,	content
0,	# FOX 新闻和媒体及政治中的关键人物...
1, ...
```

...

# 全局模式
----分析师3----
重要性得分：100
唐纳德·J·特朗普：经常与他的政治活动一起讨论...
...
`````

你可以将该上下文集成到你自定义的提示中。

</details>

<details>
<summary>提示</summary>

`nano-graphrag` 使用 `nano_graphrag.prompt.PROMPTS` 字典对象中的提示。你可以随意使用它并替换里面的任何提示。

一些重要的提示：

- `PROMPTS["entity_extraction"]` 用于从文本块中提取实体和关系。
- `PROMPTS["community_report"]` 用于组织和总结图集群的描述。
- `PROMPTS["local_rag_response"]` 是本地搜索生成的系统提示模板。
- `PROMPTS["global_reduce_rag_response"]` 是全局搜索生成的系统提示模板。
- `PROMPTS["fail_response"]` 是当没有任何内容与用户查询相关时的回退响应。

</details>

<details>
<summary>自定义分块</summary>

`nano-graphrag` 允许你自定义自己的分块方法，查看 [示例](./examples/using_custom_chunking_method.py)。

切换到内置的文本分割器分块方法：

```python
from nano_graphrag._op import chunking_by_seperators

GraphRAG(...,chunk_func=chunking_by_seperators,...)
```

</details>

<details>
<summary>LLM函数</summary>

在 `nano-graphrag` 中，我们需要的是两种类型的LLM，一个强大的和一个便宜的。前者用于规划和响应，后者用于总结。默认情况下，强大的是 `gpt-4o`，便宜的是 `gpt-4o-mini`

你可以实现你自己的LLM函数（参考 `_llm.gpt_4o_complete`）：

```python
async def my_llm_complete(
    prompt, system_prompt=None, history_messages=[], **kwargs
) -> str:
  #
 如果有的话，弹出缓存KV数据库
  hashing_kv: BaseKVStorage = kwargs.pop("hashing_kv", None)
  # 其余的kwargs用于调用LLM，例如，`max_tokens=xxx`
	...
  # 你的LLM调用
  response = await call_your_LLM(messages, **kwargs)
  return response
```

用你的替换默认的：

```python
# 根据需要调整最大令牌大小或最大异步请求
GraphRAG(best_model_func=my_llm_complete, best_model_max_token_size=..., best_model_max_async=...)
GraphRAG(cheap_model_func=my_llm_complete, cheap_model_max_token_size=..., cheap_model_max_async=...)
```

你可以查看这个 [示例](./examples/using_deepseek_as_llm.py) 使用 [`deepseek-chat`](https://platform.deepseek.com/api-docs/) 作为LLM模型

你可以查看这个 [示例](./examples/using_ollama_as_llm.py) 使用 [`ollama`](https://github.com/ollama/ollama) 作为LLM模型

#### Json输出

`nano-graphrag` 将使用 `best_model_func` 输出JSON，参数为 `"response_format": {"type": "json_object"}`。然而，有些开源模型可能产生不稳定的JSON。

`nano-graphrag` 引入了一个后处理接口，用于将响应转换为JSON。这个函数的签名如下：

```python
def YOUR_STRING_TO_JSON_FUNC(response: str) -> dict:
  "将字符串响应转换为JSON"
  ...
```

并通过 `GraphRAG(...convert_response_to_json_func=YOUR_STRING_TO_JSON_FUNC,...)` 传递你自己的函数。

例如，你可以查看 [json_repair](https://github.com/mangiucugna/json_repair) 来修复LLM返回的JSON字符串。
</details>

<details>
<summary>嵌入函数</summary>

你可以用任何 `_utils.EmbedddingFunc` 实例替换默认的嵌入函数。

例如，默认情况下，它使用OpenAI嵌入API：

```python
@wrap_embedding_func_with_attrs(embedding_dim=1536, max_token_size=8192)
async def openai_embedding(texts: list[str]) -> np.ndarray:
    openai_async_client = AsyncOpenAI()
    response = await openai_async_client.embeddings.create(
        model="text-embedding-3-small", input=texts, encoding_format="float"
    )
    return np.array([dp.embedding for dp in response.data])
```

用你的替换默认嵌入函数：

```python
GraphRAG(embedding_func=your_embed_func, embedding_batch_num=..., embedding_func_max_async=...)
```

你可以查看一个 [示例](./examples/using_local_embedding_model.py) 使用 `sentence-transformer` 在本地计算嵌入。
</details>

<details>
<summary>存储组件</summary>

你可以将所有与存储相关的组件替换为你自己的实现，`nano-graphrag` 主要使用三种存储：

**`base.BaseKVStorage` 用于存储键-JSON对的数据**

- 默认情况下，我们使用磁盘文件存储作为后端。
- `GraphRAG(.., key_string_value_json_storage_cls=YOURS,...)`

**`base.BaseVectorStorage` 用于索引嵌入**

- 默认情况下，我们使用 [`nano-vectordb`](https://github.com/gusye1234/nano-vectordb) 作为后端。
- 我们还有一个内置的 [`hnswlib`](https://github.com/nmslib/hnswlib) 存储，查看这个 [示例](./examples/using_hnsw_as_vectorDB.py)。
- 查看这个 [示例](./examples/using_milvus_as_vectorDB.py) 实现了 [`milvus-lite`](https://github.com/milvus-io/milvus-lite) 作为后端（Windows不可用）。
- `GraphRAG(.., vector_db_storage_cls=YOURS,...)`

**`base.BaseGraphStorage` 用于存储知识图谱**

- 默认情况下，我们使用 [`networkx`](https://github.com/networkx/networkx) 作为后端。
- `GraphRAG(.., graph_storage_cls=YOURS,...)`

你可以查看 `nano_graphrag.base` 了解每个组件的详细接口。
</details>

## 常见问题解答

查看 [常见问题解答](./docs/FAQ.md)。

## 路线图

查看 [路线图](./docs/ROADMAP.md)。

## 贡献

`nano-graphrag` 对任何形式的贡献都是开放的。在你贡献之前，请阅读 [这个](./docs/CONTRIBUTING.md)。

## 基准测试

- [英文基准测试](./docs/benchmark-en.md)
- [中文基准测试](./docs/benchmark-zh.md)
- [一个评估](./examples/benchmarks/eval_naive_graphrag_on_multi_hop.ipynb) 笔记本在 [多跳RAG任务](https://github.com/yixuantt/MultiHop-RAG) 上

## 问题

- `nano-graphrag` 没有实现 `GraphRAG` 的 `covariates` 功能
- `nano-graphrag` 实现的全局搜索与原始的不同。原始的使用类似map-reduce的风格将所有社区填充到上下文中，而 `nano-graphrag` 只使用最重要的和中心的社区（使用 `QueryParam.global_max_consider_community` 控制，默认为512个社区）。

---

请注意，由于原文中的一些链接和代码片段在翻译中可能无法直接访问，因此在实际使用时可能需要进行适当的调整。
