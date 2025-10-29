import os 
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing_extensions import TypedDict
from langchain_community.tools import WikipediaQueryRun, ArxivQueryRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import add_messages, START, END, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from langchain_core.messages import BaseMessage
from langchain.tools import tool

load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
os.environ['TAVILY_API_KEY'] = os.getenv("TAVILY_API_KEY")
os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGCHAIN_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")
os.environ['LANGGRAPH_API_KEY'] = os.getenv("LANGCHAIN_API_KEY")

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def super_bot_runner():

    wiki_wrapper = WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=250)
    arxiv_wrapper = ArxivAPIWrapper(top_k_results=2, doc_content_chars_max=250)

    wiki = WikipediaQueryRun(api_wrapper=wiki_wrapper)
    arxiv = ArxivQueryRun(api_wrapper=arxiv_wrapper)
    tavily = TavilySearchResults()

    ### custom tools
    @tool
    def add(a:int, b:int)->int:
        """ add a and b
        Args:
            a: first int
            b: second int
        """
        return a + b

    @tool
    def subtract(a:int, b:int)->int:
        """ subtract a and b
        Args:
            a: first int
            b: second int
        """
        return a - b

    @tool
    def multiply(a:int, b:int)->int:
        """ multiply a and b
        Args:
            a: first int
            b: second int
        """
        return a * b

    @tool
    def divide(a:int, b:int)->int:
        """ divide a and b
        Args:
            a: first int
            b: second int
        """
        return a / b

    tools = [wiki, arxiv, tavily, add, subtract, multiply, divide]
    llm = ChatGroq(model="openai/gpt-oss-120b", api_key=groq_api_key)
    llm = llm.bind_tools(tools)

    def super_bot(state:State):
        return {"messages":[llm.invoke(state["messages"])]}
    
    def should_continue(state: State):
        if state["messages"][-1].tool_calls:
            return "Tools"
        else:
            return END

    graph_builder = StateGraph(State)
    graph_builder.add_node("Agent", super_bot)
    graph_builder.add_node("Tools", ToolNode(tools))

    graph_builder.add_edge("Tools", "Agent")
    graph_builder.add_edge(START, "Agent")
    graph_builder.add_conditional_edges("Agent", should_continue)
    

    graph = graph_builder.compile()

    return graph

agent = super_bot_runner()

# langgraph dev