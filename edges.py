from agentstate import AgentState

def should_escalate(state: AgentState) -> str:
    """决定是否需要人工介入"""
    if state.get('needs_human_intervention', False):
        return "human_agent"
    else:
        return "generate_response"