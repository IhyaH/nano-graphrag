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

question = "变压器在运行状态下，相连出线刀闸和相连接地刀闸还有相应保护装置应该分别处于什么状态?"

# 打开一个文件用于写入，如果文件不存在则创建
with open('output.md', 'w', encoding='utf-8') as file:
    # 在终端和文件中输出问题
    print("Question:", question)
    file.write(f"Question: {question}\n\n")
    
    # 执行全局搜索并同时输出到终端和文件
    global_result = graph_func.query(question)
    print("Global Search Result:")
    print(global_result)
    file.write("Global Search Result:\n")
    file.write(global_result + "\n\n")
    
    # 执行局部搜索并同时输出到终端和文件
    local_result = graph_func.query(question, param=QueryParam(mode="local"))
    print("Local Search Result:")
    print(local_result)
    file.write("Local Search Result:\n")
    file.write(local_result + "\n")

print("已将输出保存到output.md文件中。")