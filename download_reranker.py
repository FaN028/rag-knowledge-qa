# -*- coding: utf-8 -*-
# 下载 rerank（重排）模型 bge-reranker-base，走国内 ModelScope 镜像
from modelscope import snapshot_download

model_dir = snapshot_download(
    "BAAI/bge-reranker-base",
    local_dir="./models/bge-reranker-base",
)
print("下载完成，rerank 模型在：", model_dir)
