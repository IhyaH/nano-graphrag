import os
# 设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL
os.environ['OPENAI_API_KEY'] = 'sk-Fc8qfHHDIxvkiDL717474aF8170243Be8a60168c7eEf6091'
os.environ['OPENAI_BASE_URL'] = 'https://openkey.cloud/v1'
# 导入语句：从nano_graphrag模块中导入GraphRAG类和QueryParam类
# nano_graphrag是自定义模块名，GraphRAG是图数据库操作类，QueryParam是查询参数配置类
from nano_graphrag import GraphRAG, QueryParam

# 创建GraphRAG类的实例，命名为graph_func
# working_dir参数指定了工作目录，这里使用了相对路径"./dickens"，意味着当前目录下的dickens文件夹
graph_func = GraphRAG(working_dir="./dickens")

# 使用print函数打印执行局部图数据库查询的结果
# 查询语句同样是"What are the top themes in this story?"，但这次指定了局部搜索模式
# QueryParam(mode="local")创建了一个查询参数实例，设置搜索模式为"local"，即局部搜索
# 局部搜索通常更精确，可扩展性更好
# 全局搜索为global
print(graph_func.query("这个故事的主题是什么?", param=QueryParam(mode="local")))