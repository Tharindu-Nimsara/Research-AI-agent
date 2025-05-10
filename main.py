from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool

load_dotenv()

#we get repsonse templates from BaseModel of Pydantic 
#this is a schema
class ResearchResponse(BaseModel):
    topic:str
    summary: str
    sources: list[str]
    tools_used: list[str] 


#choosing the llm

#this is anthropic (claude) llm
my_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# if you want to use gpt model instead of claude all you have to do is input openai api key in .env file and uncomment below code line and also comment above code line
# my_llm = ChatOpenAI(model="gpt-3.5-turbo")
my_parser = PydanticOutputParser(pydantic_object=ResearchResponse)



my_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=my_parser.get_format_instructions())
#this format_instruction consist of response template which we created above and now we pass it to the above prompt . 

my_tools = [search_tool, wiki_tool, save_tool] # These are the tools that we have connected to LLM to make an AI agent

#making an agent by feeding llm, prompt, tools
my_agent = create_tool_calling_agent(
    llm=my_llm,
    prompt=my_prompt,
    tools=my_tools
)

agent_executer = AgentExecutor(agent=my_agent, tools=my_tools ,verbose=True)

# Human input
query = input("What can i help you research? ")

# Running the agent and getting the response
raw_response = agent_executer.invoke({"query": query})

try:
    structured_respose = my_parser.parse(raw_response.get("output")[0]["text"])
    print(structured_respose)
except Exception as e:
    print("Error in parsing response",e,"Raw response - ", raw_response)    






