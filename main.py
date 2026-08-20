import os
import certifi
import requests
from dotenv import load_dotenv, find_dotenv

from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.tools import tool
from langchain.agents import create_react_agent, AgentExecutor

# ==============================================================
# LOAD ENV VARIABLES from .env file
# ==============================================================
os.environ["SSL_CERT_FILE"] = certifi.where()
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
WEATHERSTACK_API_KEY = os.getenv("WEATHERSTACK_API_KEY")

# ==============================================================
# TOOLS DEFINITION
# ==============================================================

search_tool = TavilySearchResults(max_results=2)


@tool
def get_weather_data(city: str) -> str:
    """
    Get the current weather for a given city.
    """
    url = f"http://api.weatherstack.com/current?access_key={WEATHERSTACK_API_KEY}&query={city}"
    response = requests.get(url)
    data = response.json()
    print(data)
    if "current" in data:
        return (
            f"City: {city}\n"
            f"Temperature: {data['current']['temperature']}°C\n"
            f"Weather Description: {data['current']['weather_descriptions'][0]}\n"
        )
    else:
        return f"Could not retrieve weather data for {city}. Please check the city name and try again."


# ==============================================================
# LLM SETUP
# ==============================================================
llm = ChatOpenAI(
    model="gpt-4.1-nano",
    temperature=1,
    openai_api_key=OPENAI_API_KEY
)

# ==============================================================
# PROMPT
# ==============================================================
prompt = hub.pull("hwchase17/react")

# ==============================================================
# TOOLS SETUP
# ==============================================================
tools = [search_tool, get_weather_data]

# ==============================================================
# CREATE AGENT
# ==============================================================
agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# ==============================================================
# EXECUTOR
# ==============================================================
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True
)


def main():
    # ==============================================================
    # RUN
    # ==============================================================
    response = agent_executor.invoke({
        "input": "What is latest news on USA IRAN war and what is the current weather in Tehran?"
    })
    print(response["output"])


if __name__ == "__main__":
    main()