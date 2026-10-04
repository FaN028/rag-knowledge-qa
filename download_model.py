# -*- coding: utf-8 -*-
# 下载中文 embedding 模型 bge-small-zh-v1.5
#
# 为什么走 ModelScope（魔搭）而不是 HuggingFace？
#   国内直接下载 HuggingFace 会被墙，ModelScope 是阿里的国内镜像，速度快。
#
# 用法（在项目目录下执行）：
#   1. pip install modelscope        （装一次下载工具）
#   2. python download_model.py       （运行本脚本，自动下载模型到 models/ 文件夹）
#
# 下载完成后，你的项目里会有 models/bge-small-zh-v1.5 这个文件夹，
# 里面就是 embedding 模型，rag.py 会自动从那里加载。

from modelscope import snapshot_download

model_dir = snapshot_download(
    "AI-ModelScope/bge-small-zh-v1.5",       # 模型在 ModelScope 上的名字
    local_dir="./models/bge-small-zh-v1.5",  # 下载到项目里的哪个文件夹
)
print("下载完成，模型在：", model_dir)
