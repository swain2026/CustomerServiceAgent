import uuid
from agentstate import AgentState
from workflow import build_customer_agent_graph
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage

class CustomerServiceAgent:
    """Maintains a conversation session with a specific user."""
    
    def __init__(self, user_id: str = ""):
        """Initialize a new session with optional user_id."""
        self.user_id = user_id if user_id else str(uuid.uuid4())
        self.user_info: dict = {}
        self.customer_agent = build_customer_agent_graph()
        self.system_prompt = ""
    
    def __del__(self):
        """Cleanup resources when the session is released."""
        if hasattr(self, 'customer_agent'):
            del self.customer_agent
    
    def process_input(self, user_input: str) -> AgentState:
        """Process user input and return agent state."""
        # Build initial state with existing session context
        initial_state: AgentState = {
            "messages":[HumanMessage(content=user_input)],
            "intent": "",
            "emotion": "",
            "user_id": self.user_id,
            "user_info": {},
            "order_info": {},
            "needs_human_intervention": False            
        }
        
        # Run the agent
        try:
            final_state = self.customer_agent.invoke(initial_state)
        except Exception as e:
            error_message = f"Sorry, I encountered an error processing your request: {str(e)}"
            # Log the error for debugging (you can add proper logging later)
            print(f"Agent error: {e}")
            raise Exception(error_message)
        
        # Update session state with results
        # self.user_info = result.get("user_info", self.user_info)
        # self.context_info = result.get("context_info", self.context_info)
        # self.conversation_history.append(f"User: {user_input}")
        # self.conversation_history.append(f"Agent: {result['response']}")
        
        return final_state