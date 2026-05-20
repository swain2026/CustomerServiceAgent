from typing import Optional, List, Dict, Any
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END

from agentstate import AgentState
from model import thinking_model

from tools import (
    get_latest_order,
    get_user_info, 
    retrieve_knowledge_base,
    get_product_info
 )

tools = [get_user_info, get_latest_order, retrieve_knowledge_base, get_product_info]


def classify_intent(state: AgentState) -> dict:

    """意图识别节点"""

    system_prompt = f"""
你是一个意图分类器。分析用户输入的意图，只输出以下六个选项之一，不要输出任何其他文字或解释：

咨询类
投诉类
售后类
订单类
技术支持
其他

输出格式：只输出分类结果，不要标点符号。
""".strip()

    # print("prompt:", prompt)

    # 创建一个提示模板，明确要求模型进行思考
    chat_prompt_template  = ChatPromptTemplate.from_messages([
        ("system", system_prompt)
    ])

    # Use classification_model for more deterministic output
    from model import classification_model
    response = classification_model.invoke(chat_prompt_template.format_messages())

    intent = response.content.strip()
    
    # Post-process: extract only the valid intent category
    valid_intents = ["咨询类", "投诉类", "售后类", "订单类", "技术支持", "其他"]
    for valid_intent in valid_intents:
        if valid_intent in intent:
            intent = valid_intent
            break
    else:
        # If no valid intent found, default to "其他"
        intent = "其他"

    state["intent"] = intent

    print("intent:", intent)

    return {"messages": [response]}


def analyze_emotion(state: AgentState) -> dict:

    """情绪分析节点"""

    system_prompt = f"""
你是一个情绪分类器。分析用户输入的情绪状态，只输出以下四个选项之一，不要输出任何其他文字或解释：

积极
中性
消极
愤怒

输出格式：只输出一个词，不要标点符号。
""".strip()

      # 创建一个提示模板，明确要求模型进行思考
    chat_prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"), # 占位符，用于插入历史消息
    ])
    
    # Use classification_model for more deterministic output
    from model import classification_model
    response = classification_model.invoke(chat_prompt_template.format_messages(messages=state['messages']))

    emotion = response.content.strip()
    
    # Post-process: extract only the valid emotion category
    valid_emotions = ["积极", "中性", "消极", "愤怒"]
    for valid_emotion in valid_emotions:
        if valid_emotion in emotion:
            emotion = valid_emotion
            break
    else:
        # If no valid emotion found, default to "中性"
        emotion = "中性"

    state["emotion"] = emotion

    print("emotion:", emotion)
    
    return {"messages": [response]}

def generate_response(state: AgentState):
    """
    Agent节点: 接收状态,调用带有提示的模型,返回新消息。
    """    
    # 获取用户输入
    user_input = ""
    first_msg = state['messages'][0]
    if hasattr(first_msg, 'content'):
        user_input = first_msg.content
    elif isinstance(first_msg, dict):
        user_input = first_msg.get("content", "")

    # 使用 f-string 直接构建系统提示，避免模板格式化问题
    system_prompt = f"""
你是一名智能电商客服助手，需要及时回答用户关于产品的咨询。

**当前上下文信息：**
- 用户ID: {state['user_id']}
- 用户意图: {state['intent']}
- 用户输入: {user_input}

**产品信息：**
- 你可以通过调用 get_product_info 工具查询产品价格和促销信息
- 支持的产品包括：无线蓝牙耳机、智能手表、笔记本电脑支架、降噪耳机、无线充电器等

**重要指示：**
1. 当需要查询用户信息、订单信息或产品信息时，请调用相应的工具函数
2. 如果问题与具体产品或订单相关，先调用工具获取信息再回答
3. 如果问题可以基于常识直接回答，无需调用工具
4. 你必须用中文回复最终结果
""".strip()

    # 绑定工具到模型
    model_with_tools = thinking_model.bind_tools(tools)
    
    # 构建消息列表：系统提示 + 对话历史
    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加对话历史（排除第一条消息，因为已经作为user_input使用了）
    for msg in state['messages']:
        if hasattr(msg, 'content'):
            messages.append({"role": "user", "content": msg.content})
        elif isinstance(msg, dict):
            messages.append(msg)
    
    # 调用模型
    response = model_with_tools.invoke(messages)
    
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
    print("outputs:", outputs)
    return {"messages": outputs}

# --- 5. 定义边缘条件 (Edge Condition) ---
def should_continue(state):
    """
    检查最后一条消息是否包含工具调用。
    返回 "tools" 如果需要调用工具，否则返回 "end"。
    """
    messages = state['messages']
    last_message = messages[-1]
    # 检查是否有 tool_calls 属性且不为空
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        print(f"Tool calls detected: {last_message.tool_calls}")
        return "tools"
    print("No tool calls, ending workflow")
    return "end"


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