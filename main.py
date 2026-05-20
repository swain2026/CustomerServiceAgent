"""Main entry point for the Customer Service Agent."""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize model first (must be after load_dotenv)
from model import thinking_model

from agent import CustomerServiceAgent
from agentstate import AgentState


def get_user_input() -> str:
    """Get input from the user."""
    return input("You: ")


def display_response(state: AgentState) -> None:
    """Display the agent's response."""
    for i, msg in enumerate(state['messages']):
        print(f"[{i}] {msg.type}: {msg.content}")
        if hasattr(msg, 'tool_calls'):
            print(f"    Tool Calls: {msg.tool_calls}")


def main():
    """Main function to run the customer service agent."""
    api_key = os.getenv("MODEL_API_KEY")
    base_url = os.getenv("MODEL_BASE_URL")
    model_name = os.getenv("MODEL_NAME")
    
    if not api_key:
        print("Error: MODEL_API_KEY not found in environment variables.")
        print("Please copy .env.example to .env and configure your API key.")
        sys.exit(1)
    
    if not base_url:
        print("Error: MODEL_BASE_URL not found in environment variables.")
        print("Please configure MODEL_BASE_URL in your .env file.")
        sys.exit(1)
    
    if not model_name:
        print("Error: MODEL_NAME not found in environment variables.")
        print("Please configure MODEL_NAME in your .env file.")
        sys.exit(1)
    
    print("Customer Service Agent initialized successfully!")
    print("Type your message and press Enter to get a response.")
    print("Press Ctrl+C to exit.\n")
    
    # Create a new session (in a real app, you might load user_id from auth)
    agent = CustomerServiceAgent()
    
    while True:
        try:
            user_input = get_user_input()
            if user_input.strip():
                final_state = agent.process_input(user_input)
                display_response(final_state)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)
        except EOFError:
            print("\nGoodbye!")
            sys.exit(0)


if __name__ == "__main__":
    main()


# 测试示例
# test_inputs = [
#     "我想了解一下你们的产品价格",
#     "这个智能手表多少钱，有什么优惠活动呢",
#     "我买的商品有问题，要求退货",
#     "这个产品怎么这么差劲，我要投诉！",
#     "我的订单什么时候能发货？"
# ]
