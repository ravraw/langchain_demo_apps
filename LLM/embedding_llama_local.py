from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

embed = OllamaEmbeddings(
    model="llama3.2"
)

input_text = "The meaning of life is 42"
vector = embed.embed_query(input_text)
print(vector[:3])



