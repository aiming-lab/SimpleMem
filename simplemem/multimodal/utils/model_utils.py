"""
模型名称工具函数
"""

def normalize_model_name(model_name: str) -> str:
    """
    规范化模型名称，确保使用正确的格式。
    对于标准 OpenAI API，直接返回原始模型名称。
    Bosch 代理映射已禁用，因为我们直接调用 OpenAI。
    """
    if not model_name:
        return "gpt-4o-mini"
    
    # 直接返回原始名称（适用于标准 OpenAI API）
    return model_name
