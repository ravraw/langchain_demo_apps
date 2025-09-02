
from langchain_openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

llm = OpenAI(
    model="gpt-3.5-turbo-instruct",
    temperature=0,
    max_retries=2
    # api_key="...",
    # base_url="...",
    # organization="...",
    # other params...
)

input_text = "What RAG stands for in Agentic AI space"
response = llm.invoke(input_text)
print(response)