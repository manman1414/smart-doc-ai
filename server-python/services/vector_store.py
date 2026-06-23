import chromadb
import os
import numpy as np

_CHROMA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "chroma_db")
client = chromadb.PersistentClient(path=_CHROMA_PATH)
collection = client.get_or_create_collection(name="documents")

def search_similar(query: str, doc_id: str = None, top_k: int = 3) -> list[str]:
    from .embedder import embed_text

    query_embedding = embed_text(query)

    if doc_id:
        get_result = collection.get(
            where={"doc_id": {"$eq": doc_id}},
            include=["embeddings", "documents"]
        )
        if not get_result or not get_result.get("ids"):
            return []

        chunk_embeddings = np.array(get_result["embeddings"])
        chunk_docs = get_result.get("documents") or []

        query_vec = np.array(query_embedding)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        chunk_norms = chunk_embeddings / (np.linalg.norm(chunk_embeddings, axis=1, keepdims=True) + 1e-10)
        similarities = np.dot(chunk_norms, query_norm)

        if len(similarities) == 0:
            return []
        top_indices = np.argsort(similarities)[::-1][:min(top_k, len(similarities))]
        return [chunk_docs[i] for i in top_indices if similarities[i] > 0]
    else:
        results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
        return results['documents'][0] if results.get('documents') and results['documents'] else []

def delete_doc_vectors(doc_id: str) -> None:
    """取消解析时清理已写入的半成品向量"""
    try:
        collection.delete(where={"doc_id": doc_id})
    except Exception as e:
        print(f"[VECTOR] delete_doc_vectors failed doc_id={doc_id}: {e}", flush=True)

def get_doc_text(doc_id: str, max_chars: int = 8000) -> str:
    """按 chunk_index 拼接文档全文（供摘要重试，上传临时文件已删除）"""
    result = collection.get(
        where={"doc_id": {"$eq": doc_id}},
        include=["documents", "metadatas"],
    )
    if not result or not result.get("ids"):
        return ""

    metas = result.get("metadatas") or []
    docs = result.get("documents") or []
    pairs = [
        (meta.get("chunk_index", 0) if meta else 0, doc)
        for meta, doc in zip(metas, docs)
        if doc
    ]
    pairs.sort(key=lambda x: x[0])
    full = "\n".join(text for _, text in pairs)
    return full[:max_chars] if max_chars else full