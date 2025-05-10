from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser

load_dotenv()

#we get repsonse templates from BaseModel of Pydantic 
#this is a schema
class ResearchResponse(BaseModel):
    topic:str
    summary: str
    sources: list[str]
    tools_used: list[str] 

#choosing the llm
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
parser = PydanticOutputParser(pydantic_object=ResearchResponse)

