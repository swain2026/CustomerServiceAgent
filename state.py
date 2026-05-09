
from typing import TypedDict, List, Annotated
import operator

class CustomerServiceState(TypedDict):
    # 用户输入
    user_input: str

    # 用户ID
    user_id: str
    
    # 用户信息
    user_info: dict
    
    # 订单ID（可选）
    order_id: str
    
    # 对话历史
    conversation_history: List[str]
    
    # 客服响应
    response: str
    
    # 问题分类
    intent: str
    
    # 情绪分析结果
    emotion: str
    
    # 是否需要人工介入
    needs_human_intervention: bool
    
    # 上下文信息
    context_info: dict
    
    # 订单信息
    order_info: dict