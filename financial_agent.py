from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(
    api_key = os.environ.get("GROQ_API_KEY")
)

# Created a new agent for web search
web_search_agent = Agent(
    name = "Web Search Agent",
    role = "Search the web for information",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [DuckDuckGo()],
    instructions  = ["Always include the source of the information in the notes"],
    show_tools_calls = True,
    markdown = True
)

# Created a new agent for financial data
financial_agent = Agent(
    name = "Financial Agent",
    model = Groq(id = "llama-3.3-70b-versatile"),
    tools = [
        YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news = True)
    ],
    instructions  = ["Use tables to display the data", "Always include the source of the information in the notes"],
    show_tools_calls = True,
    markdown = True
)

# Created a multi-agent
multi_agent = Agent(
    model = Groq(id = "llama-3.3-70b-versatile"),
    team = [web_search_agent, financial_agent],
    instructions = ["Always include the source of the information in the notes", "Use tables to display the data"],
    show_tools_calls = True,
    markdown = True
)

multi_agent.print_response("Summarize analyst recommendations and share latest news for AAPL", stream = True)