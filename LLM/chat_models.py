from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=100,
    timeout=None,
    max_retries=2,
    # api_key="...",  # if you prefer to pass api key in directly instaed of using env vars
    # base_url="...",
    # organization="...",
    # other params...
)

response = model.invoke("What does RAV means in the Agentic AI space?")
print(response)