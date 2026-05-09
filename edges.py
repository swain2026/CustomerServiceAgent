from state import CustomerServiceState

def should_escalate(state: CustomerServiceState) -> str:
    """决定是否需要人工介入"""
    if state.get('needs_human_intervention', False):
        return "human_agent"
    else:
        return "generate_response"