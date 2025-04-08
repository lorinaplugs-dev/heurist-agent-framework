import asyncio
import os

from heurist_core.clients.search_client import SearchClient
from heurist_core.components import LLMProvider
from heurist_core.tools.tools_mcp import Tools
from heurist_core.workflows import ResearchWorkflow

print("Successfully imported heurist_core modules!")
search_client = SearchClient(client_type="firecrawl", api_key=os.getenv("FIRECRAWL_KEY"), rate_limit=1)
tools = Tools()
llm_provider = LLMProvider(tool_manager=tools)
research_workflow = ResearchWorkflow(llm_provider, tools, search_client)

server_url = "https://localhost:8000/sse"


async def main():
    result = await research_workflow.process(message="latest bitcoin news?")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
