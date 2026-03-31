from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="meta-llama/Llama-3.2-1B-Instruct",
    task="text-generation",
    max_new_tokens=512,
    temperature=0.7
)

model = ChatHuggingFace(llm = llm)

prompt1 = PromptTemplate(
    template = 'Generate a report on the current state of {topic}.',
    input_variables = ['topic']
)

prompt2 = PromptTemplate(
    template='Based on the report, list 5 key challenges and 5 potential solutions related to {topic}.',
    input_variables = ['topic']
)

parser = StrOutputParser()

chain = prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic': 'Hindu Phobia'})

print(result)