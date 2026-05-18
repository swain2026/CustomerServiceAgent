
from typing import TypedDict, List, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    # 问题分类
    intent: str
    # 情绪分析结果
    emotion: str
    # 用户ID
    user_id: str
    # 用户信息
    user_info: dict
     # 是否需要人工介入
    needs_human_intervention: bool
    # 订单信息
    order_info: dict