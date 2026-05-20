from langchain_core.tools import tool
from typing import Optional
from pinecone import search_pinecone

# 模拟产品数据库
PRODUCTS = {
    "无线蓝牙耳机": {
        "product_name": "无线蓝牙耳机",
        "price": 299.00,
        "original_price": 399.00,
        "promotion": {
            "active": True,
            "type": "折扣",
            "value": "7.5折",
            "description": "限时优惠，满299减50"
        },
        "stock": 500,
        "rating": 4.5
    },
    "智能手表": {
        "product_name": "智能手表",
        "price": 899.00,
        "original_price": 1299.00,
        "promotion": {
            "active": True,
            "type": "满减",
            "value": "满1000减200",
            "description": "VIP用户额外9折"
        },
        "stock": 150,
        "rating": 4.8
    },
    "笔记本电脑支架": {
        "product_name": "笔记本电脑支架",
        "price": 159.00,
        "original_price": 199.00,
        "promotion": {
            "active": False,
            "type": "无",
            "value": "无",
            "description": "无促销活动"
        },
        "stock": 300,
        "rating": 4.3
    },
    "降噪耳机": {
        "product_name": "降噪耳机",
        "price": 1299.00,
        "original_price": 1599.00,
        "promotion": {
            "active": True,
            "type": "折扣",
            "value": "8折",
            "description": "新品首发，限量前100名送保护套"
        },
        "stock": 200,
        "rating": 4.7
    },
    "无线充电器": {
        "product_name": "无线充电器",
        "price": 129.00,
        "original_price": 199.00,
        "promotion": {
            "active": True,
            "type": "满减",
            "value": "满199减50",
            "description": "买二送一活动进行中"
        },
        "stock": 1000,
        "rating": 4.4
    }
}

# 模拟订单列表
ORDERS = [
    {
        "order_id": "ORD20240501001",
        "order_status": "shipped",
        "order_date": "2024-05-01",
        "estimated_delivery": "2024-05-10",
        "items": [
            {"product_name": "无线蓝牙耳机", "quantity": 1, "price": 299.00}
        ],
        "total_amount": 299.00
    },
    {
        "order_id": "ORD20240415001",
        "order_status": "delivered",
        "order_date": "2024-04-15",
        "estimated_delivery": "2024-04-20",
        "items": [
            {"product_name": "智能手表", "quantity": 1, "price": 899.00}
        ],
        "total_amount": 899.00
    },
    {
        "order_id": "ORD20240320001",
        "order_status": "delivered",
        "order_date": "2024-03-20",
        "estimated_delivery": "2024-03-25",
        "items": [
            {"product_name": "笔记本电脑支架", "quantity": 2, "price": 159.00}
        ],
        "total_amount": 318.00
    }
]

@tool
def get_user_info(user_id: str) -> dict:

    """验证用户ID并获取用户信息的组合节点"""
    result = {}

    if not user_id or user_id == "":
        return result
    

    # 获取用户信息

    # 这里可以连接真实的用户数据库或CRM系统

    # 模拟获取用户信息

    user_info = {

        "user_id": user_id,

        "registration_date": "2023-01-15",

        "last_purchase_date": "2024-03-20",

        "purchase_count": 15,

        "total_spent": 12500.00,

        "vip_level": "Gold",

        "contact_info": {

            "email": f"user_{user_id[:8]}@example.com",

            "phone": "+86-138-0000-0000"

        },

        "preferences": ["电子产品", "快速配送", "优惠活动"],

        "support_tickets": 3,

        "satisfaction_score": 4.2

    }


    print("user_id:", user_id)
    

    result["user_info"] = user_info
    
    return result

@tool
def get_latest_order(user_id: str) -> dict:

    """获取订单信息(模拟订单API)"""

    result = {}

    if not user_id or user_id == "":
        return result
    
    # 返回最新的订单（按order_date降序排列后的第一个）
    latest_order = max(ORDERS, key=lambda x: x['order_date']) if ORDERS else None

    print(f"获取订单信息: 最新订单 {latest_order['order_id'] if latest_order else '无'}")

    return {"order_info": latest_order}

@tool
def retrieve_knowledge_base(intent: str, user_input: str) -> dict:

    """从知识库检索相关信息"""

    # 只有咨询类、投诉类、技术支持需要搜索Pinecone

    knowledge_context = ""

    pinecone_results = []
    

    if intent in ['咨询类', '投诉类', '技术支持']:

        # 从Pinecone向量数据库搜索相关信息

        pinecone_result = search_pinecone_wrapper(user_input)

        pinecone_results = pinecone_result.get('search_results', [])
        

        # 构建知识库上下文

        if intent == '咨询类':

            knowledge_context = f"根据您的咨询，我们提供以下信息：[产品/服务详细信息]。VIP用户可享受特殊优惠。"

        elif intent == '投诉类':

            knowledge_context = f"投诉处理流程：[受理、调查、解决步骤]。VIP用户将获得专属客服经理处理。"

        elif intent == '技术支持':

            knowledge_context = "技术支持指南：[常见问题解决方案]"
        

        # 如果Pinecone有搜索结果，将其添加到上下文中

        if pinecone_results:

            pinecone_context = "\n\n从知识库检索到的相关信息：\n"

            for i, result in enumerate(pinecone_results, 1):

                pinecone_context += f"{i}. [相似度: {result['score']:.2f}] {result.get('text', 'N/A')}\n"

            knowledge_context += pinecone_context  
    else:
        knowledge_context = "常见问题解答：[通用FAQ信息]"

    return {

        "context_info": {

            "knowledge_context": knowledge_context,

            "retrieved_faq": f"相关FAQ: [{intent}相关常见问题]"    

        }

    }


def search_pinecone_wrapper(query: Optional[str] = None, top_k: int = 5) -> dict:
    """
    从Pinecone向量数据库搜索相关信息(包装函数,用于nodes.py调用)
    Args:
        state: 客户服务状态
        query: 搜索查询，如果未提供则使用用户输入
        top_k: 返回结果数量 
    Returns:
        包含搜索结果的字典
    """

    # 获取嵌入模型（如果已定义）

    embedding_model = globals().get('embedding_model')
    

    result = search_pinecone(
        query=query,
        top_k=top_k,
        embedding_model=embedding_model
    )   

    # 格式化为nodes.py期望的返回格式

    formatted_results = result.get('search_results', [])    

    return {
        "search_results": formatted_results,
        "context_info": {
            "knowledge_context": f"从知识库检索到 {len(formatted_results)} 条相关信息",
            "retrieved_results": formatted_results
        },
        "error": result.get('error')
    }


@tool
def get_product_info(product_name: str) -> dict:
    """根据产品名称获取产品价格和促销信息(模拟产品API)"""

    # 查找产品信息
    product_key = product_name.strip()
    product_info = PRODUCTS.get(product_key)

    if not product_info:
        # 如果找不到精确匹配，尝试模糊匹配
        for key, info in PRODUCTS.items():
            if product_key in key or key in product_key:
                product_info = info
                break

    if not product_info:
        return {
            "error": f"未找到产品: {product_name}",
            "suggestions": list(PRODUCTS.keys())
        }

    return {"product_info": product_info}
