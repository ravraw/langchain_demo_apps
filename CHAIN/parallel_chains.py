from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

model2 = ChatAnthropic(model_name="claude-3-7-sonnet-20250219")

prompt1 = PromptTemplate(
    template="Generate short and simple notes from the following text \n {text}",
    input_variables=["text"],
)

prompt2 = PromptTemplate(
    template="Generate 5 short question answers from the following text \n {text}",
    input_variables=["text"],
)

prompt3 = PromptTemplate(
    template="Merge the provided notes and quiz into a single document \n notes -> {notes} and quiz -> {quiz}",
    input_variables=["notes", "quiz"],
)

parser = StrOutputParser()

parallel_chain = RunnableParallel(
    {"notes": prompt1 | model1 | parser, "quiz": prompt2 | model2 | parser}
)

merge_chain = prompt3 | model1 | parser

chain = parallel_chain | merge_chain

text = """
In machine learning, a tensor is a mathematical object that generalizes the concept of scalars, vectors, and matrices to higher dimensions. You can think of a scalar as a single number (0D), a vector as a one-dimensional array of numbers (1D), and a matrix as a two-dimensional array (2D). A tensor extends this idea to any number of dimensions (nD), making it a flexible way to represent and organize data. For example, a color image is typically stored as a 3D tensor with dimensions representing height, width, and color channels.

Tensors are central in machine learning because they provide a uniform way to represent different types of data and perform computations on them. In deep learning frameworks like TensorFlow and PyTorch, almost all data—whether it's images, text, or audio—is stored and manipulated as tensors. Operations such as addition, multiplication, reshaping, and slicing can be applied across entire tensors efficiently, often taking advantage of parallel processing on GPUs.

Beyond just being containers of data, tensors are tightly linked to the mathematical operations that power machine learning models. Neural networks, for instance, use tensors to represent both inputs and the parameters (weights and biases) of the model. During training, tensors flow through the network, are transformed by matrix multiplications and nonlinear functions, and produce outputs. Gradients, which are also tensors, are computed during backpropagation to update the parameters. This consistent tensor-based representation allows machine learning systems to scale across complex architectures and massive datasets.
"""

result = chain.invoke({"text": text})

print(result)

chain.get_graph().print_ascii()
