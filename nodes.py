from state import CustomerServiceState

from model import thinking_model

import uuid

import os

from typing import Optional, List, Dict, Any


from pinecone import search_pinecone


def classify_intent(state: CustomerServiceState) -> dict:

    """意图识别节点"""

    prompt = f"""
    分析用户输入的意图，分类为以下几种之一：
    - 咨询类：询问产品、服务或政策
    - 投诉类：表达不满或问题
    - 售后类：退货、维修、换货等
    - 订单类：查询订单状态、物流、支付等
    - 技术支持：解决技术问题
    - 其他：无法明确分类
    用户输入：{state['user_input']}
    请直接输出分类结果，不要添加其他文字。
    """

    # print("prompt:", prompt)


    response = thinking_model.invoke(prompt)

    intent = response.content.strip()


    print("intent:", intent)
    
    # 保留现有的对话历史
    existing_history = state.get('conversation_history', [])
    
    return {

        "intent": intent,

        "conversation_history": existing_history

    }


def analyze_emotion(state: CustomerServiceState) -> dict:

    """情绪分析节点"""

    prompt = f"""
    分析用户输入的情绪状态，输出以下之一：
    - 积极
    - 中性
    - 消极
    - 愤怒
    用户输入：{state['user_input']}
    请直接输出情绪类别，不要添加其他文字。
    """

    response = thinking_model.invoke(prompt)

    emotion = response.content.strip()


    print("emotion:", emotion)
    

    return {"emotion": emotion}


def generate_response(state: CustomerServiceState) -> dict:

    """生成响应节点"""

    # 根据意图和情绪生成合适的响应

    context = f"""
    意图：{state['intent']}
    情绪：{state['emotion']}
    用户输入：{state['user_input']}
    """
    
    # 添加对话历史到上下文
    conversation_history = state.get('conversation_history', [])
    if conversation_history:
        history_text = "\n".join(conversation_history[-5:])  # 只取最近5条以控制token
        context += f"\n\n对话历史(最近5条):\n{history_text}"

    # 如果有订单信息，将其加入上下文

    if state.get('order_info'):

        order_context = f"""

        订单信息：{state['order_info']}
        """

        context += order_context
    
    # 如果有上下文信息（知识库检索结果等），将其加入
    if state.get('context_info'):
        context_info = state['context_info']
        if context_info.get('knowledge_context'):
            context += f"\n\n知识库信息：\n{context_info['knowledge_context']}"
    

    if state['emotion'] == '愤怒':

        # 针对愤怒用户采用安抚策略

        prompt = f"""

        用户情绪激动，需要耐心安抚并解决问题。
        

        {context}
        

        请生成一个安抚性的回复，承认用户的问题并承诺会解决。
        """

    elif state['intent'] == '投诉类':

        # 针对投诉类问题采用专业处理方式

        prompt = f"""
        用户提出投诉，需要专业处理。
        {context}
        请生成一个专业的回复，承认问题并说明处理流程。
        """
    else:

        # 一般咨询类回复

        prompt = f"""
        {context} 
        请生成一个专业、友好的回复来解答用户问题。
        """
    
    # print("prompt:", prompt)

    response = thinking_model.invoke(prompt)
    

    # 判断是否需要人工介入

    needs_human = False

    if state['emotion'] in ['愤怒', '消极'] and state['intent'] in ['投诉类', '技术支持']:

        # 情绪激动且涉及复杂问题时建议人工介入

        needs_human = True

    elif len(state['user_input']) > 500:  # 输入过长可能包含复杂问题

        needs_human = True
    

    return {

        "response": response.content,

        "needs_human_intervention": needs_human

    }


def escalate_to_human(state: CustomerServiceState) -> dict:

    """人工介入节点"""

    escalation_message = """

    已将您的问题转接至人工客服，稍后会有专员为您处理。

    我们已记录您的问题详情,无需重复描述。

    请您保持通话,客服专员将在1分钟内接通。
    """
    

    return {

        "response": escalation_message,

        "needs_human_intervention": True

    }


def update_conversation_history(state: CustomerServiceState) -> dict:

    """更新对话历史"""

    new_history = state.get('conversation_history', [])

    new_history.append(f"用户: {state['user_input']}")

    new_history.append(f"客服: {state['response']}")
    
    # 限制历史长度以控制上下文窗口
    if len(new_history) > 10:
        new_history = new_history[-10:]

    print("add history:", len(new_history))
    
    return {"conversation_history": new_history}


def retrieve_knowledge_base(state: CustomerServiceState) -> dict:

    """从知识库检索相关信息"""

    user_vip_level = state['user_info'].get('vip_level', 'Regular')

    order_info = state.get('order_info', {})

    latest_order = order_info.get('latest_order', {})
    

    # 只有咨询类、投诉类、技术支持需要搜索Pinecone

    knowledge_context = ""

    pinecone_results = []
    

    if state['intent'] in ['咨询类', '投诉类', '技术支持']:

        # 从Pinecone向量数据库搜索相关信息

        pinecone_result = search_pinecone_wrapper(state)

        pinecone_results = pinecone_result.get('search_results', [])
        

        # 构建知识库上下文

        if state['intent'] == '咨询类':

            knowledge_context = f"根据您的咨询，我们提供以下信息：[产品/服务详细信息]。VIP用户 {user_vip_level} 可享受特殊优惠。"

        elif state['intent'] == '投诉类':

            knowledge_context = f"投诉处理流程：[受理、调查、解决步骤]。VIP用户 {user_vip_level} 将获得专属客服经理处理。"

        elif state['intent'] == '技术支持':

            knowledge_context = "技术支持指南：[常见问题解决方案]"
        

        # 如果Pinecone有搜索结果，将其添加到上下文中

        if pinecone_results:

            pinecone_context = "\n\n从知识库检索到的相关信息：\n"

            for i, result in enumerate(pinecone_results, 1):

                pinecone_context += f"{i}. [相似度: {result['score']:.2f}] {result.get('text', 'N/A')}\n"

            knowledge_context += pinecone_context

    elif state['intent'] == '订单类':

        order_status = latest_order.get('status', 'N/A')

        tracking_num = latest_order.get('tracking_number', 'N/A')

        estimated_delivery = latest_order.get('estimated_delivery', 'N/A')
        

        knowledge_context = f"""

        订单信息：

        - 订单号：{latest_order.get('order_id', 'N/A')}

        - 当前状态：{order_status}

        - 物流单号：{tracking_num}

        - 预计送达：{estimated_delivery}

        - 配送地址：{latest_order.get('shipping_address', 'N/A')}

        - 支付方式：{latest_order.get('payment_method', 'N/A')}
        """
    else:

        knowledge_context = "常见问题解答：[通用FAQ信息]"
    

    return {

        "context_info": {

            "knowledge_context": knowledge_context,

            "retrieved_faq": f"相关FAQ: [{state['intent']}相关常见问题]",

            "product_info": f"产品信息：[{state['intent']}涉及的相关产品/服务]",

            "user_priority": user_vip_level,

            "latest_order": latest_order,

            "pinecone_results": pinecone_results,

            "pinecone_result_count": len(pinecone_results)

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


def get_user_info(state: CustomerServiceState) -> dict:

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



def get_order_info(state: CustomerServiceState) -> dict:

    """获取订单信息(模拟订单API)"""

    user_id = state.get('user_id')
    

    # 模拟从订单API获取订单信息

    # 实际应用中这里会调用真实的订单服务API

    order_id = state.get('order_id')
    

    if order_id:

        # 根据订单ID获取特定订单

        order_info = {

            "order_id": order_id,

            "user_id": user_id,

            "order_status": "shipped",

            "order_date": "2024-05-01",

            "estimated_delivery": "2024-05-10",

            "items": [

                {

                    "product_name": "无线蓝牙耳机",

                    "product_id": "PROD001",

                    "quantity": 1,

                    "price": 299.00,

                    "sku": "SKU123456"

                }

            ],

            "total_amount": 299.00,

            "shipping_address": {

                "recipient": "张三",

                "address": "北京市朝阳区XX路XX号",

                "zip_code": "100000",

                "phone": "13800138000"

            },

            "tracking_number": "YT123456789CN",

            "carrier": "顺丰速运"

        }
    else:

        # 获取用户最近的订单列表

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
    

    print(f"获取订单信息: {order_id if order_id else '最近订单列表'}")
    

    return {"order_info": order_info}



