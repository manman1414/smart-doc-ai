from sentence_transformers import SentenceTransformer

# 全局加载一次模型（首次会自动下载，约 400MB，之后缓存到本地）
# BGE 中文小模型，效果好且轻量，CPU 也能跑
model = SentenceTransformer("BAAI/bge-small-zh-v1.5")

def embed_text(text: str) -> list[float]:
    """
    将单段文本转换为向量（浮点数列表）
    """
    return model.encode(text).tolist()

def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    批量将多段文本转换为向量
    """
    embeddings = model.encode(texts)
    return [e.tolist() for e in embeddings]