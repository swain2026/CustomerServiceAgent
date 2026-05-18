from langchain_core.tools import tool
from pinecone import search_pinecone

@tool
def get_user_info(user_id: str) -> dict:

    """验证用户ID并获取用户信息的组合节点"""
    result = {}
    
    user_id = state.get('user_id')

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
def get_order_info(user_id: str) -> dict:

    """获取订单信息(模拟订单API)"""

    user_id = state.get('user_id')
    
    # 模拟获取用户最近的订单列表

    order_info = {

        "orders": [

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

        ],

        "total_orders": 3
    }    

    print(f"获取订单信息: 最近订单列表")    

    return {"order_info": order_info}

@tool
def retrieve_knowledge_base() -> dict:

    """从知识库检索相关信息"""  

    # 只有咨询类、投诉类、技术支持需要搜索Pinecone

    knowledge_context = ""

    pinecone_results = []
    

    if state['intent'] in ['咨询类', '投诉类', '技术支持']:

        # 从Pinecone向量数据库搜索相关信息

        pinecone_result = search_pinecone_wrapper(state)

        pinecone_results = pinecone_result.get('search_results', [])
        

        # 构建知识库上下文

        if state['intent'] == '咨询类':

            knowledge_context = f"根据您的咨询，我们提供以下信息：[产品/服务详细信息]。VIP用户可享受特殊优惠。"

        elif state['intent'] == '投诉类':

            knowledge_context = f"投诉处理流程：[受理、调查、解决步骤]。VIP用户将获得专属客服经理处理。"

        elif state['intent'] == '技术支持':

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

            "retrieved_faq": f"相关FAQ: [{state['intent']}相关常见问题]"       

        }

    }


def search_pinecone_wrapper(state: CustomerServiceState, query: Optional[str] = None, top_k: int = 5) -> dict:
    """

    从Pinecone向量数据库搜索相关信息（包装函数，用于nodes.py调用）
    

    Args:

        state: 客户服务状态

        query: 搜索查询，如果未提供则使用用户输入

        top_k: 返回结果数量
        

    Returns:

        包含搜索结果的字典
    """

    search_query = query if query else state.get('user_input', '')
    

    # 获取嵌入模型（如果已定义）

    embedding_model = globals().get('embedding_model')
    

    result = search_pinecone(

        query=search_query,

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
