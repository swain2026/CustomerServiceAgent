from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from agentstate import AgentState
from model import thinking_model

from tools import (
    get_order_info,
    get_user_info, 
    retrieve_knowledge_base
 )

tools = [get_user_info, get_order_info, retrieve_knowledge_base]


def classify_intent(state: AgentState) -> dict:

    """意图识别节点"""

    system_prompt = f"""
    分析用户输入的意图，分类为以下几种之一：
    - 咨询类：询问产品、服务或政策
    - 投诉类：表达不满或问题
    - 售后类：退货、维修、换货等
    - 订单类：查询订单状态、物流、支付等
    - 技术支持：解决技术问题
    - 其他：无法明确分类    
    请直接输出分类结果，不要添加其他文字。
    """.strip()

    # print("prompt:", prompt)

    # 创建一个提示模板，明确要求模型进行思考
    chat_prompt_template  = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ])

    response = thinking_model.invoke(chat_prompt_template.format_messages())

    intent = response.content.strip()

    print("intent:", intent)

    return {"messages": [response]}


def analyze_emotion(state: AgentState) -> dict:

    """情绪分析节点"""

    system_prompt = f"""
    分析用户输入的情绪状态，输出以下之一：
    - 积极
    - 中性
    - 消极
    - 愤怒    
    请直接输出情绪类别，不要添加其他文字。
    """.strip()

      # 创建一个提示模板，明确要求模型进行思考
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"), # 占位符，用于插入历史消息
    ])
    
    response = thinking_model.invoke(chat_prompt_template.format_messages(messages=state['messages']))

    emotion = response.content.strip()


    print("emotion:", emotion)
    
    return {"messages": [response]}

def generate_response(state: AgentState):
    """
    Agent节点：接收状态，调用带有提示的模型，返回新消息。
    """
    system_prompt = f"""
    你是一名智能电商客服助手，需要及时回答用户关于产品的咨询。如果问题与具体产品无关，可以直接回答。

    请以 `Answer: [你的答案]` 的格式输出最终结果。

    **示例：**
    ```
    Answer: 还有其他可以帮您的吗？
    ```

    如果涉及具体产品信息，你需要以 **Thought（思考）→ Action（行动）→ Observation（观察）** 的循环方式处理问题：
    - 使用 **Thought** 描述你的分析过程
    - 使用 **Action** 调用可用工具之一，然后等待 **Observation**
    - 当你有最终答案时，以 `Answer: [你的答案]` 格式输出

    **可用工具：**
    1. `get_user_info(user_id: str)`：验证用户ID并获取用户信息，包括注册日期、最近购买日期、购买次数、总消费金额、VIP等级、联系方式、偏好设置、工单数量和满意度评分
    2. `get_order_info(user_id: str)`：获取用户的订单信息，包括订单ID、订单状态、订单日期、预计送达时间、商品列表（名称、数量、价格）和订单总金额
    3. `retrieve_knowledge_base()`：从知识库检索相关信息，根据用户意图（咨询类、投诉类、技术支持）提供对应的知识内容和FAQ信息

    **使用 Action 时，必须按以下格式格式化：**
    ```
    Action: 工具名称: 参数
    ```

    **示例流程：**
    ```
    Thought: 我需要查询用户ID为12345的订单信息
    Action: get_order_info: 12345

    Observation: {"order_info": {"orders": [{"order_id": "ORD20240501001", "order_status": "shipped", ...}], "total_orders": 3}}

    Thought: 我现在已获得回答问题所需的全部信息
    Answer: 您最近有3个订单，最新订单ORD20240501001已发货，预计5月10日送达。
    ```

    **注意：你必须用中文回复最终结果**
    
    现在轮到你了：
    """.strip()

    # 创建一个提示模板，明确要求模型进行思考
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"), # 占位符，用于插入历史消息
    ])

    model_with_tools_and_prompt = prompt | thinking_model.bind_tools(tools) # 使用管道操作符链接prompt和model

    # 将整个消息历史传递给经过提示模板处理的模型链
    response = model_with_tools_and_prompt.invoke({"messages": state["messages"]})
    
    # 处理模型的响应，无论是工具调用还是直接回复
    return {"messages": [response]}


def tool_execution_node(state):
    """
    工具执行节点：接收包含工具调用的消息，执行工具并返回结果。
    """
    messages = state['messages']
    last_message = messages[-1]
    
    outputs = []
    for tool_call in last_message.tool_calls:
        tool_result = tool_executor.invoke(tool_call)
        outputs.append(
            ToolMessage(
                content=str(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"]
            )
        )
    return {"messages": outputs}

# --- 5. 定义边缘条件 (Edge Condition) ---
def should_continue(state):
    """
    检查最后一条消息是否包含工具调用。
    """
    messages = state['messages']
    last_message = messages[-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return END


def generate_response2(state: AgentState):

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


def escalate_to_human(state: AgentState):

    """人工介入节点"""

    escalation_message = """

    已将您的问题转接至人工客服，稍后会有专员为您处理。

    我们已记录您的问题详情,无需重复描述。

    请您保持通话,客服专员将在1分钟内接通。
    """

    state.messages.append({"role": "assistant", "content": escalation_message})
    
    return END