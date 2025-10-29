import os 
from dotenv import load_dotenv
# from pydantic import BaseModel
from langgraph.graph import add_messages
from typing import Annotated
from typing_extensions import TypedDict
# from langchain_core.messages import AnyMessage
from langchain_groq import ChatGroq
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import BaseMessage

load_dotenv()
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv('GROQ_API_KEY')


class State(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]

llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)

def make_default_graph():
    graph_workflow=StateGraph(State)

    def call_model(state):
        return {"messages":[llm.invoke(state['messages'])]}
    
    graph_workflow.add_node("agent", call_model)
    graph_workflow.add_edge("agent", END)
    graph_workflow.add_edge(START, "agent")

    agent=graph_workflow.compile()
    return agent

agent = make_default_graph()
