from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal

load_dotenv()


llm = HuggingFaceEndpoint(
    repo_id="MiniMaxAI/MiniMax-M2.5",
    task="text-generation"
)

model = ChatHuggingFace(llm = llm)


parser1 = StrOutputParser()

class FeedbackSentiment(BaseModel):
    sentiment: Literal['positive', 'negative', 'neutral'] = Field(description="Give the sentiment of the feedback")

parser2 = PydanticOutputParser(pydantic_object=FeedbackSentiment)

prompt1 = PromptTemplate(
    template = 'Classify the sentiment of the following feedback as positive, negative, or neutral: \n {feedback} \n {format_instructions}',
    input_variables = ['feedback'], 
    partial_variables = {'format_instructions': parser2.get_format_instructions()}
)

classifer_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template='Respond with an appropriate response to the user for this positive feedback: \n {feedback} \n {sentiment}',
    input_variables = ['feedback', 'sentiment']
)

prompt3 = PromptTemplate(
    template='Respond with an appropriate response to the user for this negative feedback: \n {feedback} \n {sentiment}', 
    input_variables = ['feedback', 'sentiment']
)

prompt4 = PromptTemplate(
    template='Respond with an appropriate response to the user for this neutral feedback: \n {feedback} \n {sentiment}',
    input_variables = ['feedback', 'sentiment']
)

branch_chain = RunnableBranch(
    (lambda input: input["sentiment"] == 'positive', prompt2 | model | parser1),
    (lambda input: input["sentiment"] == 'negative', prompt3 | model | parser1), 
    (lambda input: input["sentiment"] == 'neutral', prompt4 | model | parser1), 
    RunnableLambda(lambda input: "Invalid sentiment")
)

chain = (
    lambda x: {
        "feedback": x["feedback"],
        "sentiment": classifer_chain.invoke(x).sentiment
    }
) | branch_chain

result = chain.invoke({'feedback': 'The product quality is terrible and I am very dissatisfied with my purchase!'})

print(result)