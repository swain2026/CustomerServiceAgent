import os
from typing import Optional, List, Dict, Any


def get_pinecone_client():
    """获取Pinecone客户端实例"""
    try:
        from pinecone import Pinecone
    except ImportError:
        raise ImportError("请安装pinecone包: pip install pinecone-client")
    
    api_key = os.getenv('PINECONE_API_KEY')
    if not api_key:
        raise ValueError("PINECONE_API_KEY未配置，请在.env文件中设置PINECONE_API_KEY")
    
    return Pinecone(api_key=api_key)


def get_pinecone_index(index_name: Optional[str] = None):
    """获取Pinecone索引实例"""
    pc = get_pinecone_client()
    index_name = index_name or os.getenv('PINECONE_INDEX_NAME', 'customer-service-kb')
    return pc.Index(index_name)


def search_pinecone(
    query: str,
    index_name: Optional[str] = None,
    top_k: int = 5,
    embedding_model = None
) -> Dict[str, Any]:
    """
    从Pinecone向量数据库搜索相关信息
    
    Args:
        query: 搜索查询文本
        index_name: 索引名称，如果未提供则使用环境变量
        top_k: 返回结果数量
        embedding_model: 嵌入模型，用于将查询转换为向量
        
    Returns:
        包含搜索结果的字典
    """
    try:
        index = get_pinecone_index(index_name)
    except (ImportError, ValueError) as e:
        return {
            "search_results": [],
            "error": str(e)
        }
    
    # 获取嵌入向量
    if embedding_model:
        try:
            query_vector = embedding_model.embed_query(query)
        except Exception as e:
            print(f"嵌入模型生成向量失败: {str(e)}")
            query_vector = None
    else:
        query_vector = None
    
    # 如果没有嵌入向量，生成模拟向量
    if not query_vector:
        print("使用模拟向量（实际应用中应使用真实的嵌入模型）")
        query_vector = [0.1] * 1536  # 假设是1536维的OpenAI嵌入
    
    try:
        # 执行搜索
        search_results = index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )
        
        # 格式化搜索结果
        formatted_results = []
        for match in search_results.get('matches', []):
            formatted_results.append({
                "id": match.get('id'),
                "score": match.get('score', 0),
                "metadata": match.get('metadata', {}),
                "text": match.get('metadata', {}).get('text', '')
            })
        
        print(f"Pinecone搜索完成,找到 {len(formatted_results)} 个结果")
        
        return {
            "search_results": formatted_results,
            "result_count": len(formatted_results)
        }
        
    except Exception as e:
        print(f"Pinecone搜索出错: {str(e)}")
        return {
            "search_results": [],
            "error": str(e)
        }
