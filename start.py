import os
# 设置环境变量 OPENAI_API_KEY 和 OPENAI_BASE_URL
os.environ['OPENAI_API_KEY'] = '***'
os.environ['OPENAI_BASE_URL'] = '***'
os.environ['PYTHONIOENCODING'] = 'UTF-8'
# 导入语句：从nano_graphrag模块中导入GraphRAG类和QueryParam类
# nano_graphrag是自定义模块名，GraphRAG是图数据库操作类，QueryParam是查询参数配置类

import logging

# 配置日志输出到文件
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='app.log',  # 日志文件路径
    filemode='a'  # 追加模式
)

from nano_graphrag import GraphRAG, QueryParam

# 创建GraphRAG类的实例，命名为graph_func
# working_dir参数指定了工作目录，这里使用了相对路径"./dickens"，意味着当前目录下的dickens文件夹
graph_func = GraphRAG(working_dir="./dickens")

# 使用with语句打开"./book.txt"文件，确保文件操作完成后自动关闭文件
# with语句是上下文管理器，可以自动处理文件打开和关闭
with open("./book.txt", encoding="utf-8") as f:
    # 调用graph_func实例的insert方法，读取并插入文件内容到图数据库中
    # f.read()是文件对象的读方法，读取文件的全部内容
    graph_func.insert(f.read())

# 使用print函数打印执行图数据库查询的结果
# 查询语句是"What are the top themes in this story?"，执行全局搜索
# 查询结果通过graph_func实例的query方法获取
print(graph_func.query("What are the top themes in this story?"))

# 使用print函数打印执行局部图数据库查询的结果
# 查询语句同样是"What are the top themes in this story?"，但这次指定了局部搜索模式
# QueryParam(mode="local")创建了一个查询参数实例，设置搜索模式为"local"，即局部搜索
# 局部搜索通常更精确，可扩展性更好
print(graph_func.query("What are the top themes in this story?", param=QueryParam(mode="local")))