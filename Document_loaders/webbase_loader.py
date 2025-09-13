from langchain_community.document_loaders import WebBaseLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

prompt = PromptTemplate(
    template="Answer the following question \n {question} from the following text - \n {text}",
    input_variables=["question", "text"],
)

parser = StrOutputParser()

url = "https://www.walmart.ca/en/flyer?icid=home_page_other_weekly_flyer_d_128753_7Y5EU5KV4E"
""
loader = WebBaseLoader(url)

docs = loader.load()

print(docs[0].page_content)


chain = prompt | model | parser

print(
    chain.invoke(
        {
            "question": "What is the prodcut that we are talking about?",
            "text": docs[0].page_content,
        }
    )
)
