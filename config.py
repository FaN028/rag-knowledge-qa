# ============================================================
# config.py —— 所有"常量"集中在这里，方便统一修改
# ============================================================
from pathlib import Path

APP_TITLE = "AI智能伴侣"
SESSIONS_DIR = Path("sessions")
LOGO_PATH = Path("resources/WLive48x48.png")

# 默认昵称 / 性格（想改默认值，只改这里一处就行）
DEFAULT_NICK_NAME = "玥玥"
DEFAULT_NATURE = "活泼开朗的姑娘"

DEEPSEEK_API_KEY_ENV = "DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
MODEL_NAME = "deepseek-v4-pro"

# —— 知识库（切块）参数 ——
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50
TOP_K = 4

# —— 模型路径 ——
MODEL_PATH = "models/bge-small-zh-v1.5"        # embedding 模型
RERANK_MODEL_PATH = "models/bge-reranker-base" # rerank 模型
RERANK_TOP_N = 20                               # 粗排先取多少个候选
