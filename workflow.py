from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode

from agentstate import AgentState

from nodes import (
    classify_intent,
    analyze_emotion,
    generate_response,
    escalate_to_human,
    should_continue
)

from edges import should_escalate

from tools import (
    get_order_info,
    get_user_info, 
    retrieve_knowledge_base
 )

tools = [get_user_info, get_order_info, retrieve_knowledge_base]

def build_customer_agent_graph():

    """构建智能客服工作流"""

    workflow = StateGraph(AgentState)

    # 添加节点
    workflow.add_node("classify_intent", classify_intent)

    workflow.add_node("analyze_emotion", analyze_emotion)

    workflow.add_node("generate_response", generate_response)

    workflow.add_node("tools", ToolNode(tools))

    workflow.add_node("human_agent", escalate_to_human)

    # 设置边界
    workflow.set_entry_point("classify_intent")    

    # 意图分类后，识别情绪
    workflow.add_edge("classify_intent", "analyze_emotion")

    
    # 情绪分析后判断是否需要人工介入
    workflow.add_conditional_edges(
        "analyze_emotion",
        should_escalate,
        {
            "human_agent": "human_agent",
            "generate_response": "generate_response"
        }
    )

    workflow.add_conditional_edges(
        "generate_response",
        should_continue,
        {
            "tools": "tools",
            END: END
        }
    )
    workflow.add_edge("tools", "generate_response")


    return workflow.compile()