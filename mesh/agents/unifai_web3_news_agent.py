import logging
import os
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class UnifWeb3NewsAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("UNIFAI_API_KEY")
        if not self.api_key:
            raise ValueError("UNIFAI_API_KEY environment variable is required")

        self.api_endpoint = "https://backend.unifai.network/api/v1/actions/call"
        self.web3news_id = 11

        self.metadata.update(
            {
                "name": "UnifAI Web3 News Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent fetches the latest Web3 and cryptocurrency news using UnifAI's API",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query for news retrieval",
                        "type": "str",
                        "required": False,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, return only raw data without natural language response",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Natural language summary of the latest Web3 news",
                        "type": "str",
                    },
                    {
                        "name": "data",
                        "description": "Structured data containing Web3 news articles",
                        "type": "dict",
                    },
                ],
                "external_apis": ["UnifAI"],
                "tags": ["Web3 News"],
                "recommended": True,
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Unifai.png",
                "examples": [
                    "What are the latest crypto news?",
                    "Get me the most recent Web3 developments",
                    "Show the top blockchain news",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """
        You are a Web3 news specialist that helps users stay informed about the latest developments in blockchain, cryptocurrency, and decentralized technologies.

        When presenting Web3 news:
        1. Summarize the key headlines with brief descriptions
        2. Highlight any significant market-moving events
        3. Organize information by topic when possible (DeFi, NFTs, Regulation, etc.)
        4. Provide relevant context for news items

        Remember to:
        - Be factual and objective in reporting
        - Note the recency of news items when possible
        - Avoid speculation or investment advice
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_web3_news",
                    "description": "Fetch the latest news from the Web3 and cryptocurrency space",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of news articles to return (1-5)",
                                "default": 3,
                            },
                            "keyword": {
                                "type": "string",
                                "description": "Optional keyword to filter news by",
                                "default": "",
                            },
                        },
                    },
                },
            }
        ]

    # ------------------------------------------------------------------------
    #                       API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=900)  # Cache for 15 minutes
    @with_retry(max_retries=3)
    async def get_web3_news(self, limit: int = 3, keyword: str = "") -> Dict:
        """
        Fetches the latest Web3 news from UnifAI API

        Args:
            limit: Maximum number of news articles to return (1-5)
            keyword: Optional keyword to filter news by

        Returns:
            Dict containing the news articles or error information
        """

        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        try:
            if limit < 10:
                limit = 10

            action = f"Web3News/{self.web3news_id}/getWeb3News"
            payload = {"limit": limit}

            if keyword:
                payload["keyword"] = keyword

            logger.info(f"Fetching Web3 news with params: {payload}")

            async with aiohttp.ClientSession() as session:
                data = {"action": action, "payload": payload}
                async with session.post(self.api_endpoint, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully fetched Web3 news")
                        return {"status": "success", "data": result}
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return {"status": "error", "error": f"API error: {response.status} - {error_text}"}

            return {"status": "error", "error": "Maximum retries exceeded"}

        except Exception as e:
            logger.error(f"Error fetching Web3 news: {str(e)}")
            return {"status": "error", "error": f"Failed to get Web3 news: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """Handle tool execution logic"""
        if tool_name == "get_web3_news":
            limit = function_args.get("limit", 3)
            keyword = function_args.get("keyword", "")

            logger.info(f"Getting Web3 news with limit={limit}, keyword='{keyword}'")
            result = await self.get_web3_news(limit=limit, keyword=keyword)

            errors = self._handle_error(result)
            if errors:
                return errors

            return result
        else:
            return {"error": f"Unsupported tool: {tool_name}"}

    # ------------------------------------------------------------------------
    #                       MAIN MESSAGE HANDLER
    # ------------------------------------------------------------------------
    async def _before_handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before message handling to preprocess parameters.
        Routes natural language queries to the appropriate tool.
        """
        query = params.get("query")

        if query and not params.get("tool"):
            modified_params = params.copy()
            modified_params["tool"] = "get_web3_news"

            limit = 10
            keyword = ""

            if any(term in query.lower() for term in ["many", "several", "multiple", "five", "5"]):
                limit = 5
            elif any(term in query.lower() for term in ["few", "brief", "one", "1"]):
                limit = 1

            lower_query = query.lower()
            keywords = ["bitcoin", "ethereum", "defi", "nft", "regulation", "crypto", "blockchain"]
            for kw in keywords:
                if kw in lower_query:
                    keyword = kw
                    break

            modified_params["tool_arguments"] = {"limit": limit, "keyword": keyword}

            thinking_msg = "Fetching the latest Web3 news..."
            self.push_update(params, thinking_msg)

            return modified_params

        return await super()._before_handle_message(params)
