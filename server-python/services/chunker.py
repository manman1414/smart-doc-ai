def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    将长文本按固定长度分块，块与块之间有重叠
    
    参数:
        text: 要分块的长文本
        chunk_size: 每块最大字符数（默认500）
        overlap: 相邻块之间的重叠字符数（默认50）
    
    返回:
        文本块列表
    
    示例:
        text = "这是一段很长的文档内容..."
        chunks = chunk_text(text)
        # 返回 ["这是第一段内容...", "这是第二段内容...", ...]
    """
    # 如果文本很短，不超过一块，直接返回
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        # 取当前块的结束位置
        end = start + chunk_size
        # 截取当前块
        chunk = text[start:end]
        chunks.append(chunk)
        # 移动起始位置：前进 chunk_size - overlap
        # 这样相邻两块会有 overlap 个字符的重叠
        start += chunk_size - overlap
        # 防止死循环
        if start >= len(text):
            break
    
    return chunks