# AI 智能伴侣 · RAG 问答系统

一个基于 **Streamlit + DeepSeek** 的 RAG（检索增强生成）问答系统。支持文档知识库、语义向量检索、rerank 重排和答案可溯源。

## ✨ 功能特性

- 多会话流式对话（可自定义昵称、性格的 AI 伴侣）
- 上传文档（txt / md / pdf / docx）构建私有知识库
- 语义向量检索（BGE 中文 embedding + 余弦相似度）
- Rerank 重排（bge-reranker 精排，提升检索精度）
- 答案可溯源（回答下方展示命中的知识库片段）
- 多轮对话记忆（历史发言向量化，检索相关记忆融入回答）

## 🧠 核心流程（RAG 三步）

```
① 上传文档 → 解析 → 切块 → 向量化 → 存入知识库
② 提问 → 向量化 → 相似度检索 → rerank 精排 → 挑出最相关片段
③ 片段注入提示词 → DeepSeek 生成回答 → 流式显示 + 展示引用来源
```

## 🛠 技术栈

| 层 | 技术 |
|---|---|
| 界面 | Streamlit |
| 大模型 | DeepSeek API（OpenAI SDK） |
| Embedding | BAAI/bge-small-zh-v1.5 |
| Rerank | BAAI/bge-reranker-base |
| 相似度 | numpy（余弦相似度） |

## 📊 评测结果

自建 10 题测试集，检索命中率（Top3）：

| 检索方法 | 命中率 |
|---|---|
| 关键词检索 | 80% |
| 向量检索 | **100%** |
| 向量 + rerank | **100%** |

## 🚀 安装与运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载模型（走国内 ModelScope 镜像）
python download_model.py       # 下载 embedding 模型
python download_reranker.py    # 下载 rerank 模型

# 3. 配置 DeepSeek API Key（环境变量）
#    Windows: set DEEPSEEK_API_KEY=你的key

# 4. 运行
streamlit run app.py
```

## 📁 目录结构

```
ai/
├── app.py                    # 程序入口（页面配置 + AI 调用 + 主程序）
├── config.py                 # 所有常量
├── session.py                # 会话管理
├── rag.py                    # RAG：模型加载 + 切块 + 检索 + 注入
├── ui.py                     # 界面（聊天记录、知识库面板、侧边栏）
├── evaluate.py               # 评测脚本（对比三种检索的命中率）
├── download_model.py         # 下载 embedding 模型
├── download_reranker.py      # 下载 rerank 模型
├── requirements.txt          # 依赖清单
├── models/                   # 模型文件（下载后生成）
├── sessions/                 # 会话记录（运行后生成）
└── resources/                # logo 图片
```
