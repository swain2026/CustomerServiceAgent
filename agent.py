import uuid

from state import CustomerServiceState
from workflow import build_customer_agent_graph

class CustomerServiceSession:
    """Maintains a conversation session with a specific user."""
    
    def __init__(self, user_id: str = ""):
        """Initialize a new session with optional user_id."""
        self.user_id = user_id if user_id else str(uuid.uuid4())
        self.user_info: dict = {}
        self.conversation_history: list = []
        self.context_info: dict = {}
        self.customer_agent = build_customer_agent_graph()
    
    def __del__(self):
        """Cleanup resources when the session is released."""
        if hasattr(self, 'customer_service_app'):
            del self.customer_service_app
    
    def process_input(self, user_input: str) -> str:
        """Process user input and return agent response."""
        # Build initial state with existing session context
        initial_state: CustomerServiceState = {
            "user_input": user_input,
            "user_id": self.user_id,
            "user_info": self.user_info,
            "conversation_history": self.conversation_history.copy(),
            "response": "",
            "intent": "",
            "emotion": "",
            "needs_human_intervention": False,
            "context_info": self.context_info,
            "order_info": {}
        }
        
        # Run the agent
        try:
            result = self.customer_agent.invoke(initial_state)
        except Exception as e:
            error_message = f"Sorry, I encountered an error processing your request: {str(e)}"
            # Log the error for debugging (you can add proper logging later)
            print(f"Agent error: {e}")
            return error_message
        
        # Update session state with results
        self.user_info = result.get("user_info", self.user_info)
        self.context_info = result.get("context_info", self.context_info)
        self.conversation_history.append(f"User: {user_input}")
        self.conversation_history.append(f"Agent: {result['response']}")
        
        return result["response"]