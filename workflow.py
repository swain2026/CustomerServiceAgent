from langgraph.graph import StateGraph

from state import CustomerServiceState

from nodes import (

    classify_intent,

    analyze_emotion,

    generate_response,

    escalate_to_human,

    update_conversation_history,
    get_user_info,
    get_order_info,

    retrieve_knowledge_base
)

from edges import should_escalate


def build_customer_agent_graph():

    """构建智能客服工作流"""

    workflow = StateGraph(CustomerServiceState)
    

    # 添加节点

    workflow.add_node("get_user_info", get_user_info)

    workflow.add_node("classify_intent", classify_intent)

    workflow.add_node("get_order_info", get_order_info)

    workflow.add_node("retrieve_knowledge", retrieve_knowledge_base)

    workflow.add_node("analyze_emotion", analyze_emotion)

    workflow.add_node("generate_response", generate_response)

    workflow.add_node("update_history", update_conversation_history)

    workflow.add_node("human_agent", escalate_to_human)
    

    # 设置边

    workflow.set_entry_point("get_user_info")
    

    # 确保用户ID后，进行意图分类

    workflow.add_edge("get_user_info", "classify_intent")
    

    # 意图分类后，如果是订单相关则获取订单信息

    workflow.add_conditional_edges(

        "classify_intent",

        lambda state: "order_info" if "order" in state.get("intent", "").lower() else "no_order",

        {

            "order_info": "get_order_info",

            "no_order": "analyze_emotion"

        }
    )
    

    # 获取订单信息后继续知识库检索

    workflow.add_edge("get_order_info", "retrieve_knowledge")
    

    # 意图分类后直接进入知识库检索（非订单类）

    workflow.add_edge("classify_intent", "retrieve_knowledge")
    

    # 知识库检索后继续情绪分析

    workflow.add_edge("retrieve_knowledge", "analyze_emotion")
    

    # 情绪分析后判断是否需要人工介入

    workflow.add_conditional_edges(

        "analyze_emotion",

        should_escalate,

        {

            "human_agent": "human_agent",

            "generate_response": "generate_response"

        }
    )
    

    # 生成响应后结束

    workflow.add_edge("generate_response", "__end__")

    workflow.add_edge("human_agent", "__end__")
    

    return workflow.compile()