import os
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# model configuation

thinking_model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    openai_api_key=os.getenv("MODEL_API_KEY"),
    openai_api_base=os.getenv("MODEL_BASE_URL"),
    temperature=0.7,
    max_tokens=1000
)

# Use lower temperature for classification tasks to ensure format compliance
classification_model = ChatOpenAI(
    model=os.getenv("MODEL_NAME"),
    openai_api_key=os.getenv("MODEL_API_KEY"),
    openai_api_base=os.getenv("MODEL_BASE_URL"),
    temperature=0,
    max_tokens=50
)

embedding_model = OpenAIEmbeddings(
    model=os.getenv("EM_MODEL_NAME"),
    openai_api_key=os.getenv("MODEL_API_KEY"),
    openai_api_base=os.getenv("MODEL_BASE_URL")
)