import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import requests
from smolagents import ToolCallingAgent, tool
from smolagents.memory import SystemPromptStep

from core.custom_smolagents import OpenAIServerModel
from core.llm import call_llm_async
from decorators import monitor_execution, with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)


class CoinGeckoTokenInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.public_api_url = "https://api.coingecko.com/api/v3"
        self.pro_api_url = "https://pro-api.coingecko.com/api/v3"
        self.api_key = os.getenv("COINGECKO_API_KEY")
        if not self.api_key:
            raise ValueError("COINGECKO_API_KEY environment variable is required")

        self.public_headers = {"Authorization": f"Bearer {self.api_key}"}
        self.pro_headers = {"x-cg-pro-api-key": self.api_key}

        self.metadata.update(
            {
                "name": "CoinGecko Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent can fetch token information, market data, trending coins, and category data from CoinGecko.",
                "external_apis": ["Coingecko"],
                "tags": ["Trading"],
                "recommended": True,
                "large_model_id": "google/gemini-2.0-flash-001",
                "small_model_id": "google/gemini-2.0-flash-001",
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Coingecko.png",
                "examples": [
                    "Top 5 crypto by market cap",
                    "24-hr price change of ETH",
                    "Get information about HEU",
                    "Analyze AI16Z token",
                    "List crypto categories",
                    "Compare DeFi tokens",
                    "Get trending on-chain pools",
                    "Get top token holders for a token",
                ],
            }
        )

        # Initialize SmolaAgents setup
        self.model = OpenAIServerModel(
            model_id=self.metadata["large_model_id"],
            api_key=self.heurist_api_key,
            api_base=self.heurist_base_url,
        )

        tools = [
            self.get_token_info_tool(),
            self.get_trending_coins_tool(),
            self.get_token_price_multi_tool(),
            self.get_categories_list_tool(),
            self.get_category_data_tool(),
            self.get_tokens_by_category_tool(),
            self.get_trending_pools_tool(),
            self.get_top_token_holders_tool(),
        ]

        max_steps = 6
        self.agent = ToolCallingAgent(tools=tools, model=self.model, max_steps=max_steps)

        self.agent.prompt_templates["system_prompt"] = self.get_system_prompt()
        self.agent.system_prompt = self.agent.prompt_templates["system_prompt"]
        self.agent.memory.system_prompt = SystemPromptStep(system_prompt=self.agent.system_prompt)

        self.agent.step_callbacks.append(self._step_callback)
        self.current_message = {}
        self._request_semaphore = asyncio.Semaphore(10)  # Limit to 10 concurrent requests

    def _step_callback(self, step_log):
        logger.info(f"Step: {step_log}")
        if step_log.tool_calls:
            msg = f"Calling function {step_log.tool_calls[0].name} with args {step_log.tool_calls[0].arguments}"
            logger.info(msg)
            self.push_update(self.current_message, msg)

    def get_system_prompt(self) -> str:
        return """
    IDENTITY:
    You are a crypto data specialist that can fetch token information and category data from CoinGecko.

    CAPABILITIES:
    - Search and retrieve token details
    - Get current trending coins
    - Analyze token market data
    - Compare multiple tokens using the token price multi tool
    - List crypto categories
    - Get tokens within specific categories
    - Compare tokens across categories
    - Get trending on-chain pools
    - Get top token holders for a specific token and network

    RESPONSE GUIDELINES:
    - Keep responses focused on what was specifically asked
    - Format numbers in a human-readable way (e.g., "$150.4M")
    - Provide only relevant metrics for the query context

    DOMAIN-SPECIFIC RULES:
    For specific token queries, identify whether the user provided a CoinGecko ID directly or needs to search by token name or symbol. Coingecko ID is lowercase string and may contain dashes. If a direct CoinGecko ID lookup fails, the system will automatically attempt to search and find the best matching token. Do not make up CoinGecko IDs.

    For trending coins requests, use the get_trending_coins tool to fetch the current top trending cryptocurrencies.

    For trending pools requests, use the get_trending_pools tool to fetch trending on-chain pools. The 'include' parameter must be one of: base_token, quote_token, dex, or network.

    For top token holders requests, use the get_top_token_holders tool to analyze token holder distribution for a specific token and network.

    For token comparisons or when needing to fetch multiple token prices at once, use the get_token_price_multi tool which is more efficient than making multiple individual calls.

    For category-related requests:
    - Use get_categories_list to fetch all available categories
    - Use get_category_data to get market data for all categories
    - Use get_tokens_by_category to fetch tokens within a specific category

    When selecting tokens from search results, apply these criteria in order:
    1. First priority: Select the token where name or symbol perfectly matches the query
    2. If multiple matches exist, select the token with the highest market cap rank (lower number = higher rank)
    3. If market cap ranks are not available, prefer the token with the most complete information

    For comparison queries across tokens or categories, extract the relevant metrics and provide a comparative analysis.

    IMPORTANT:
    - Never invent or assume CoinGecko IDs or category IDs
    - Keep responses concise and relevant
    - Use multiple tool calls when needed to get comprehensive information"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_token_info",
                    "description": "Get detailed token information and market data using CoinGecko ID. This tool provides comprehensive cryptocurrency data including current price, market cap, trading volume, price changes, and more.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "coingecko_id": {"type": "string", "description": "The CoinGecko ID of the token"}
                        },
                        "required": ["coingecko_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_trending_coins",
                    "description": "Get the current top trending cryptocurrencies on CoinGecko. This tool retrieves a list of the most popular cryptocurrencies based on trading volume and social media mentions.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_token_price_multi",
                    "description": "Fetch price data for multiple tokens at once using CoinGecko IDs. Efficiently retrieves current prices and optional market data for multiple cryptocurrencies in a single API call.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ids": {
                                "type": "string",
                                "description": "Comma-separated CoinGecko IDs of the tokens to query",
                            },
                            "vs_currencies": {
                                "type": "string",
                                "description": "Comma-separated target currencies (e.g., usd,eur,btc)",
                                "default": "usd",
                            },
                            "include_market_cap": {
                                "type": "boolean",
                                "description": "Include market capitalization data",
                                "default": False,
                            },
                            "include_24hr_vol": {
                                "type": "boolean",
                                "description": "Include 24hr trading volume data",
                                "default": False,
                            },
                            "include_24hr_change": {
                                "type": "boolean",
                                "description": "Include 24hr price change percentage",
                                "default": False,
                            },
                            "include_last_updated_at": {
                                "type": "boolean",
                                "description": "Include timestamp of when the data was last updated",
                                "default": False,
                            },
                            "precision": {
                                "type": "string",
                                "description": "Decimal precision for currency values (e.g., 'full' for maximum precision)",
                                "default": False,
                            },
                        },
                        "required": ["ids", "vs_currencies"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_categories_list",
                    "description": "Get a list of all available cryptocurrency categories from CoinGecko. This tool retrieves all the category IDs and names that can be used for further category-specific queries.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_category_data",
                    "description": "Get market data for all cryptocurrency categories from CoinGecko. This tool retrieves comprehensive information about all categories including market cap, volume, market cap change, top coins in each category, and more.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "order": {
                                "type": "string",
                                "description": "Sort order for categories (default: market_cap_desc)",
                                "enum": [
                                    "market_cap_desc",
                                    "market_cap_asc",
                                    "name_desc",
                                    "name_asc",
                                    "market_cap_change_24h_desc",
                                    "market_cap_change_24h_asc",
                                ],
                            }
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_tokens_by_category",
                    "description": "Get a list of tokens within a specific category. This tool retrieves token data for all cryptocurrencies that belong to a particular category, including price, market cap, volume, and price changes.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "category_id": {
                                "type": "string",
                                "description": "The CoinGecko category ID (e.g., 'layer-1')",
                            },
                            "vs_currency": {
                                "type": "string",
                                "description": "The currency to show results in (default: usd)",
                                "default": "usd",
                            },
                            "order": {
                                "type": "string",
                                "description": "Sort order for tokens (default: market_cap_desc)",
                                "enum": [
                                    "market_cap_desc",
                                    "market_cap_asc",
                                    "volume_desc",
                                    "volume_asc",
                                    "id_asc",
                                    "id_desc",
                                ],
                                "default": "market_cap_desc",
                            },
                            "per_page": {
                                "type": "integer",
                                "description": "Number of results per page (1-250, default: 100)",
                                "default": 100,
                                "minimum": 1,
                                "maximum": 250,
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number (default: 1)",
                                "default": 1,
                                "minimum": 1,
                            },
                        },
                        "required": ["category_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_trending_pools",
                    "description": "Get up to 10 trending on-chain pools with token data from CoinGecko. The 'include' parameter must be one of: base_token, quote_token, dex, or network.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "include": {
                                "type": "string",
                                "description": "Single attribute to include: base_token, quote_token, dex, or network",
                                "enum": ["base_token", "quote_token", "dex", "network"],
                                "default": "base_token",
                            },
                            "pools": {
                                "type": "integer",
                                "description": "Number of pools to return (1-10)",
                                "default": 4,
                                "minimum": 1,
                                "maximum": 10,
                            },
                        },
                        "required": [],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_top_token_holders",
                    "description": "Get top 50 token holder addresses for a token on a specific network.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "network": {"type": "string", "description": "Network ID (e.g., base)"},
                            "address": {"type": "string", "description": "Token contract address"},
                        },
                        "required": ["network", "address"],
                    },
                },
            },
        ]

    def get_api_config(self, endpoint: str) -> tuple[str, Dict[str, str]]:
        """Determine the appropriate API URL and headers based on the endpoint."""
        if endpoint.startswith("/onchain"):
            return self.pro_api_url, self.pro_headers
        return self.public_api_url, self.public_headers

    async def _api_request(self, url: str, method: str, headers: Dict, params: Dict = None) -> Dict:
        """Perform an asynchronous API request with rate-limiting."""
        async with self._request_semaphore:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.request(method, url, headers=headers, params=params, timeout=10) as response:
                        response.raise_for_status()
                        return await response.json()
            except aiohttp.ClientResponseError as e:
                logger.error(f"API request error: {e.status}, message='{e.message}', url='{url}'")
                return {"error": f"API request failed: {e.status}, message='{e.message}', url='{url}'"}
            except aiohttp.ClientError as e:
                logger.error(f"Client error during API request: {e}")
                return {"error": f"Client error: {str(e)}"}
            except asyncio.TimeoutError:
                logger.error(f"API request timed out: {url}")
                return {"error": f"Request timed out: {url}"}

    # Tool definitions using smolagents tool decorator
    def get_token_info_tool(self):
        @tool
        def get_token_info(coingecko_id: str) -> Dict[str, Any]:
            """Get detailed token information and market data using CoinGecko ID.

            Args:
                coingecko_id: The CoinGecko ID of the token

            Returns:
                Dictionary with token information or error message
            """
            logger.info(f"Getting token info for: {coingecko_id}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config(f"/coins/{coingecko_id}")
                    response = requests.get(f"{api_url}/coins/{coingecko_id}", headers=headers)
                    response.raise_for_status()
                    return self.format_token_info(response.json())
                except requests.HTTPError:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    if response.status_code != 200:
                        api_url, headers = self.get_api_config("/search")
                        search_response = requests.get(f"{api_url}/search?query={coingecko_id}", headers=headers)
                        search_response.raise_for_status()
                        search_results = search_response.json()

                        if search_results.get("coins") and len(search_results["coins"]) > 0:
                            valid_tokens = [
                                token for token in search_results["coins"] if token.get("market_cap_rank") is not None
                            ]
                            if valid_tokens:
                                valid_tokens.sort(key=lambda x: x.get("market_cap_rank", float("inf")))
                                fallback_id = valid_tokens[0]["id"]
                                api_url, headers = self.get_api_config(f"/coins/{fallback_id}")
                                fallback_response = requests.get(f"{api_url}/coins/{fallback_id}", headers=headers)
                                fallback_response.raise_for_status()
                                return self.format_token_info(fallback_response.json())

                        return {"error": "Failed to fetch token info and fallback search failed"}
                except requests.RequestException as e:
                    logger.error(f"Error getting token info: {e}")
                    return {"error": f"Failed to fetch token info: {str(e)}"}

        return get_token_info

    def get_trending_coins_tool(self):
        @tool
        def get_trending_coins() -> Dict[str, Any]:
            """Get the current top trending cryptocurrencies on CoinGecko.

            Returns:
                Dictionary with trending coins data or error message
            """
            logger.info("Getting trending coins")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/search/trending")
                    response = requests.get(f"{api_url}/search/trending", headers=headers)
                    response.raise_for_status()
                    trending_data = response.json()
                    formatted_trending = []
                    for coin in trending_data.get("coins", [])[:10]:
                        coin_info = coin["item"]
                        formatted_trending.append(
                            {
                                "name": coin_info["name"],
                                "symbol": coin_info["symbol"],
                                "market_cap_rank": coin_info.get("market_cap_rank", "N/A"),
                                "price_usd": coin_info["data"].get("price", "N/A"),
                            }
                        )
                    return {"trending_coins": formatted_trending}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting trending coins: {e}")
                    return {"error": f"Failed to fetch trending coins: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting trending coins: {e}")
                    return {"error": f"Failed to fetch trending coins: {str(e)}"}

        return get_trending_coins

    def get_token_price_multi_tool(self):
        @tool
        def get_token_price_multi(
            ids: str,
            vs_currencies: str,
            include_market_cap: bool = False,
            include_24hr_vol: bool = False,
            include_24hr_change: bool = False,
            include_last_updated_at: bool = False,
            precision: str = None,
        ) -> Dict[str, Any]:
            """Fetch price data for multiple tokens at once using the simple/price endpoint.

            Args:
                ids: Comma-separated CoinGecko IDs of the tokens to query
                vs_currencies: Comma-separated target currencies (e.g., usd,eur,btc)
                include_market_cap: Include market capitalization data
                include_24hr_vol: Include 24hr trading volume data
                include_24hr_change: Include 24hr price change percentage
                include_last_updated_at: Include timestamp of when the data was last updated
                precision: Decimal precision for currency values (e.g., 'full' for maximum precision)

            Returns:
                Dictionary with price data for the requested tokens or error message
            """
            logger.info(f"Getting multi-token price data for: {ids} in {vs_currencies}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/simple/price")
                    params = {
                        "ids": ids,
                        "vs_currencies": vs_currencies,
                        "include_market_cap": str(include_market_cap).lower(),
                        "include_24hr_vol": str(include_24hr_vol).lower(),
                        "include_24hr_change": str(include_24hr_change).lower(),
                        "include_last_updated_at": str(include_last_updated_at).lower(),
                    }

                    if precision:
                        params["precision"] = precision

                    response = requests.get(f"{api_url}/simple/price", headers=headers, params=params)
                    response.raise_for_status()
                    price_data = response.json()

                    # Format the response in a more readable structure
                    formatted_data = {}
                    for token_id, data in price_data.items():
                        formatted_data[token_id] = data

                    return {"price_data": formatted_data}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting multi-token price data: {e}")
                    return {"error": f"Failed to fetch price data: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting multi-token price data: {e}")
                    return {"error": f"Failed to fetch price data: {str(e)}"}

        return get_token_price_multi

    def get_categories_list_tool(self):
        @tool
        def get_categories_list() -> Dict[str, Any]:
            """Get a list of all available cryptocurrency categories from CoinGecko.

            Returns:
                Dictionary with categories list or error message
            """
            logger.info("Getting categories list")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/coins/categories/list")
                    response = requests.get(f"{api_url}/coins/categories/list", headers=headers)
                    response.raise_for_status()
                    return {"categories": response.json()}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting categories list: {e}")
                    return {"error": f"Failed to fetch categories list: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting categories list: {e}")
                    return {"error": f"Failed to fetch categories list: {str(e)}"}

        return get_categories_list

    def get_category_data_tool(self):
        @tool
        def get_category_data(order: str = "market_cap_desc") -> Dict[str, Any]:
            """Get market data for all cryptocurrency categories from CoinGecko.

            Args:
                order: Sort order for categories (default: market_cap_desc)

            Returns:
                Dictionary with category data or error message
            """
            logger.info(f"Getting category data with order: {order}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/coins/categories")
                    params = {}
                    if order:
                        params["order"] = order

                    response = requests.get(f"{api_url}/coins/categories", headers=headers, params=params)
                    response.raise_for_status()

                    category_data = response.json()
                    for category in category_data:
                        if "top_3_coins" in category:
                            del category["top_3_coins"]
                        if "updated_at" in category:
                            del category["updated_at"]
                        if "top_3_coins_id" in category:
                            del category["top_3_coins_id"]

                    return {"category_data": category_data}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting category data: {e}")
                    return {"error": f"Failed to fetch category data: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting category data: {e}")
                    return {"error": f"Failed to fetch category data: {str(e)}"}

        return get_category_data

    def get_tokens_by_category_tool(self):
        @tool
        def get_tokens_by_category(
            category_id: str,
            vs_currency: str = "usd",
            order: str = "market_cap_desc",
            per_page: int = 100,
            page: int = 1,
        ) -> Dict[str, Any]:
            """Get a list of tokens within a specific category.

            Args:
                category_id: The CoinGecko category ID (e.g., 'layer-1')
                vs_currency: The currency to show results in (default: usd)
                order: Sort order for tokens (default: market_cap_desc)
                per_page: Number of results per page (1-250, default: 100)
                page: Page number (default: 1)

            Returns:
                Dictionary with category tokens or error message
            """
            logger.info(f"Getting tokens for category: {category_id}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/coins/markets")
                    params = {
                        "vs_currency": vs_currency,
                        "category": category_id,
                        "order": order,
                        "per_page": per_page,
                        "page": page,
                        "sparkline": "false",
                    }
                    response = requests.get(f"{api_url}/coins/markets", headers=headers, params=params)
                    response.raise_for_status()
                    return {
                        "category_tokens": {
                            "category_id": category_id,
                            "tokens": response.json(),
                            "pagination": {
                                "total": response.headers.get("x-total", "N/A"),
                                "per_page": response.headers.get("x-per-page", "N/A"),
                            },
                        }
                    }
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting tokens for category: {e}")
                    return {"error": f"Failed to fetch tokens for category '{category_id}': {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting tokens for category: {e}")
                    return {"error": f"Failed to fetch tokens for category '{category_id}': {str(e)}"}

        return get_tokens_by_category

    def get_trending_pools_tool(self):
        @tool
        def get_trending_pools(include: str = "base_token", pools: int = 4) -> Dict[str, Any]:
            """Get trending on-chain pools from CoinGecko.

            Args:
                include: Single attribute to include: base_token, quote_token, dex, or network
                pools: Number of pools to return (1-10, default: 4)

            Returns:
                Dictionary with trending pools data or error message
            """
            valid_includes = ["base_token", "quote_token", "dex", "network"]
            if include not in valid_includes:
                logger.error(f"Invalid include parameter: {include}. Must be one of {valid_includes}")
                return {"error": f"Invalid include parameter: {include}. Must be one of {valid_includes}"}

            logger.info(f"Getting trending pools with include: {include}, pools: {pools}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config("/onchain/pools/trending_search")
                    params = {"include": include, "pools": pools}
                    response = requests.get(f"{api_url}/onchain/pools/trending_search", headers=headers, params=params)
                    response.raise_for_status()
                    return {"trending_pools": response.json()}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting trending pools: {e}")
                    return {"error": f"Failed to fetch trending pools: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting trending pools: {e}")
                    return {"error": f"Failed to fetch trending pools: {str(e)}"}

        return get_trending_pools

    def get_top_token_holders_tool(self):
        @tool
        def get_top_token_holders(network: str, address: str) -> Dict[str, Any]:
            """Get top 50 token holder addresses for a token on a specific network.

            Args:
                network: Network ID (e.g., base)
                address: Token contract address

            Returns:
                Dictionary with top token holders data or error message
            """
            logger.info(f"Getting top token holders for network: {network}, address: {address}")
            max_attempts = 3
            for attempt in range(max_attempts):
                try:
                    api_url, headers = self.get_api_config(f"/onchain/networks/{network}/tokens/{address}/top_holders")
                    url = f"{api_url}/onchain/networks/{network}/tokens/{address}/top_holders"
                    response = requests.get(url, headers=headers)
                    response.raise_for_status()
                    return {"top_holders": response.json()}
                except requests.HTTPError as e:
                    if response.status_code == 429 and attempt < max_attempts - 1:
                        logger.warning(f"429 Too Many Requests, retrying after 5 seconds (attempt {attempt + 1})")
                        time.sleep(5)
                        continue
                    logger.error(f"Error getting top token holders: {e}")
                    return {"error": f"Failed to fetch top token holders: {str(e)}"}
                except requests.RequestException as e:
                    logger.error(f"Error getting top token holders: {e}")
                    return {"error": f"Failed to fetch top token holders: {str(e)}"}

        return get_top_token_holders

    async def select_best_token_match(self, search_results: Dict, query: str) -> str:
        """
        Select best matching token using the following criteria:
        1. Ignore tokens with None market_cap_rank
        2. Find closest name/symbol match
        3. Use market cap rank as tiebreaker
        """
        if not search_results.get("coins"):
            return None

        # Filter out tokens with None market_cap_rank
        valid_tokens = [token for token in search_results["coins"] if token.get("market_cap_rank") is not None]

        if not valid_tokens:
            return None

        # Create prompt for token selection
        token_selection_prompt = f"""Given the search query "{query}" and these token results:
        {json.dumps(valid_tokens, indent=2)}

        Select the most appropriate token based on these criteria in order:
        1. Find the token where name or symbol most closely matches the query
        - Exact matches are preferred
        - For partial matches, consider string similarity and common variations
        2. If multiple tokens have similar name matches, select the one with the highest market cap rank (lower number = higher rank)

        Return only the CoinGecko ID of the selected token, nothing else."""

        selected_token = await call_llm_async(
            base_url=self.heurist_base_url,
            api_key=self.heurist_api_key,
            model_id=self.metadata["small_model_id"],
            messages=[
                {
                    "role": "system",
                    "content": "You are a token selection assistant. You only return the CoinGecko ID of the best matching token based on the given criteria.",
                },
                {"role": "user", "content": token_selection_prompt},
            ],
            temperature=0.1,
        )

        # Clean up response to get just the ID
        selected_token = selected_token.strip().strip('"').strip("'")

        # Verify the selected ID exists in filtered results
        if any(token["id"] == selected_token for token in valid_tokens):
            return selected_token
        return None

    @with_retry(max_retries=3)
    async def _search_token(self, query: str) -> str | None:
        """internal helper to search for a token and return its id"""
        try:
            api_url, headers = self.get_api_config("/search")
            url = f"{api_url}/search"
            params = {"query": query}
            response = await self._api_request(url=url, method="GET", headers=headers, params=params)
            if "error" in response:
                return None

            if not response.get("coins"):
                return None

            if len(response["coins"]) == 1:
                return response["coins"][0]["id"]

            # try exact matches first
            for coin in response["coins"]:
                if coin["name"].lower() == query.lower() or coin["symbol"].lower() == query.lower():
                    return coin["id"]

            # if no exact match, return first result
            return response["coins"][0]["id"]
        except Exception as e:
            logger.error(f"Error searching for token: {e}")
            return None

    # ------------------------------------------------------------------------
    #                      COINGECKO API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    async def _get_trending_coins(self) -> dict:
        try:
            api_url, headers = self.get_api_config("/search/trending")
            url = f"{api_url}/search/trending"
            response = await self._api_request(url=url, method="GET", headers=headers)
            if "error" in response:
                return {"error": response["error"]}

            # Format the trending coins data
            formatted_trending = []
            for coin in response.get("coins", [])[:10]:
                coin_info = coin["item"]
                formatted_trending.append(
                    {
                        "name": coin_info["name"],
                        "symbol": coin_info["symbol"],
                        "market_cap_rank": coin_info.get("market_cap_rank", "N/A"),
                        "price_usd": coin_info["data"].get("price", "N/A"),
                    }
                )
            return {"trending_coins": formatted_trending}

        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch trending coins: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    @with_retry(max_retries=3)
    async def _get_token_info(self, coingecko_id: str) -> dict:
        try:
            api_url, headers = self.get_api_config(f"/coins/{coingecko_id}")
            url = f"{api_url}/coins/{coingecko_id}"
            response = await self._api_request(url=url, method="GET", headers=headers)
            if "error" not in response:
                return response

            # if response contains error, try search fallback
            fallback_id = await self._search_token(coingecko_id)
            if fallback_id:
                api_url, headers = self.get_api_config(f"/coins/{fallback_id}")
                fallback_url = f"{api_url}/coins/{fallback_id}"
                fallback_response = await self._api_request(url=fallback_url, method="GET", headers=headers)
                if "error" not in fallback_response:
                    return fallback_response
            return {"error": "Failed to fetch token info"}
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch token info: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    @with_retry(max_retries=3)
    async def _get_categories_list(self) -> dict:
        """Get a list of all CoinGecko categories"""
        try:
            api_url, headers = self.get_api_config("/coins/categories/list")
            url = f"{api_url}/coins/categories/list"
            response = await self._api_request(url=url, method="GET", headers=headers)
            if "error" in response:
                return {"error": response["error"]}
            return {"categories": response}
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch categories list: {str(e)}"}

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def _get_category_data(self, order: Optional[str] = "market_cap_desc") -> dict:
        """Get market data for all cryptocurrency categories"""
        try:
            api_url, headers = self.get_api_config("/coins/categories")
            url = f"{api_url}/coins/categories"
            params = {}
            if order:
                params["order"] = order

            response = await self._api_request(url=url, method="GET", headers=headers, params=params)
            if "error" in response:
                return {"error": response["error"]}

            # Process the response to remove specified fields
            category_data = response
            for category in category_data:
                if "top_3_coins" in category:
                    del category["top_3_coins"]
                if "updated_at" in category:
                    del category["updated_at"]
                if "top_3_coins_id" in category:
                    del category["top_3_coins_id"]

            return {"category_data": category_data}
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch category data: {str(e)}"}

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def _get_token_price_multi(
        self,
        ids: str,
        vs_currencies: str,
        include_market_cap: bool = False,
        include_24hr_vol: bool = False,
        include_24hr_change: bool = False,
        include_last_updated_at: bool = False,
        precision: str = None,
    ) -> dict:
        try:
            api_url, headers = self.get_api_config("/simple/price")
            url = f"{api_url}/simple/price"
            params = {
                "ids": ids,
                "vs_currencies": vs_currencies,
                "include_market_cap": str(include_market_cap).lower(),
                "include_24hr_vol": str(include_24hr_vol).lower(),
                "include_24hr_change": str(include_24hr_change).lower(),
                "include_last_updated_at": str(include_last_updated_at).lower(),
            }

            if precision:
                params["precision"] = precision

            response = await self._api_request(url=url, method="GET", headers=headers, params=params)
            if "error" in response:
                return {"error": response["error"]}
            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch multi-token price data: {str(e)}"}

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def _get_tokens_by_category(
        self,
        category_id: str,
        vs_currency: str = "usd",
        order: str = "market_cap_desc",
        per_page: int = 100,
        page: int = 1,
    ) -> dict:
        """Get tokens within a specific category"""
        try:
            api_url, headers = self.get_api_config("/coins/markets")
            url = f"{api_url}/coins/markets"
            params = {
                "vs_currency": vs_currency,
                "category": category_id,
                "order": order,
                "per_page": per_page,
                "page": page,
                "sparkline": "false",
            }

            response = await self._api_request(url=url, method="GET", headers=headers, params=params)
            if "error" in response:
                return {"error": response["error"]}
            return {"category_tokens": {"category_id": category_id, "tokens": response}}
        except Exception as e:
            logger.error(f"Error: {e}")
            return {"error": f"Failed to fetch tokens for category '{category_id}': {str(e)}"}

    def format_token_info(self, data: Dict) -> Dict:
        """Format token information in a structured way"""
        market_data = data.get("market_data", {})
        return {
            "token_info": {
                "id": data.get("id", "N/A"),
                "name": data.get("name", "N/A"),
                "symbol": data.get("symbol", "N/A").upper(),
                "market_cap_rank": data.get("market_cap_rank", "N/A"),
                "categories": data.get("categories", []),
            },
            "market_metrics": {
                "current_price_usd": market_data.get("current_price", {}).get("usd", "N/A"),
                "market_cap_usd": market_data.get("market_cap", {}).get("usd", "N/A"),
                "fully_diluted_valuation_usd": market_data.get("fully_diluted_valuation", {}).get("usd", "N/A"),
                "total_volume_usd": market_data.get("total_volume", {}).get("usd", "N/A"),
            },
            "price_metrics": {
                "ath_usd": market_data.get("ath", {}).get("usd", "N/A"),
                "ath_change_percentage": market_data.get("ath_change_percentage", {}).get("usd", "N/A"),
                "ath_date": market_data.get("ath_date", {}).get("usd", "N/A"),
                "high_24h_usd": market_data.get("high_24h", {}).get("usd", "N/A"),
                "low_24h_usd": market_data.get("low_24h", {}).get("usd", "N/A"),
                "price_change_24h": market_data.get("price_change_24h", "N/A"),
                "price_change_percentage_24h": market_data.get("price_change_percentage_24h", "N/A"),
            },
            "supply_info": {
                "total_supply": market_data.get("total_supply", "N/A"),
                "max_supply": market_data.get("max_supply", "N/A"),
                "circulating_supply": market_data.get("circulating_supply", "N/A"),
            },
        }

    @monitor_execution()
    @with_retry(max_retries=3)
    async def handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Either 'query' or 'tool' is required in params.
        - If 'tool' is provided, call that tool directly with 'tool_arguments' (bypassing the LLM).
        - If 'query' is provided, route via SmolaAgents for dynamic tool selection.
        """
        query = params.get("query")
        tool_name = params.get("tool")
        tool_args = params.get("tool_arguments", {})
        raw_data_only = params.get("raw_data_only", False)
        self.current_message = params

        try:
            # ---------------------
            # 1) DIRECT TOOL CALL
            # ---------------------
            if tool_name:
                logger.info(f"Direct tool call: {tool_name} with args {tool_args}")
                result = await self._handle_tool_logic(tool_name, tool_args)

                if raw_data_only:
                    return {"response": "", "data": result}
                if query:
                    explanation = await self._respond_with_llm(
                        model_id=self.metadata["large_model_id"],
                        system_prompt=self.get_system_prompt(),
                        query=query,
                        tool_call_id="direct_tool",
                        data=result,
                        temperature=0.7,
                    )
                    return {"response": explanation, "data": result}

                # For direct tool calls without query, just return the data
                return {"response": "", "data": result}

            # ---------------------
            # 2) NATURAL LANGUAGE QUERY (using SmolaAgents)
            # ---------------------
            if query:
                logger.info(f"Processing natural language query: {query}")

                result = self.agent.run(
                    f"""Analyze this query and provide insights: {query}

                        Guidelines:
                        - Use appropriate tools to find and analyze cryptocurrency data
                        - Format numbers clearly (e.g. $1.5M, 15.2%)
                        - Keep response concise and focused on key insights
                        """
                )
                response_text = result.to_string()

                return {
                    "response": response_text,
                    "data": {},
                }

            # ---------------------
            # 3) NEITHER query NOR tool
            # ---------------------
            return {"error": "Either 'query' or 'tool' must be provided in the parameters."}

        except Exception as e:
            logger.error(f"Agent execution failed: {str(e)}")
            return {"error": str(e)}
        finally:
            self.current_message = {}

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle execution of specific tools and return the raw data"""
        if tool_name == "get_token_info":
            result = await self._get_token_info(function_args["coingecko_id"])
            if not isinstance(result, dict) or "error" not in result:
                return self.format_token_info(result)
            return result

        elif tool_name == "get_trending_coins":
            return await self._get_trending_coins()

        elif tool_name == "get_token_price_multi":
            result = await self._get_token_price_multi(
                ids=function_args["ids"],
                vs_currencies=function_args.get("vs_currencies", "usd"),
                include_market_cap=function_args.get("include_market_cap", False),
                include_24hr_vol=function_args.get("include_24hr_vol", False),
                include_24hr_change=function_args.get("include_24hr_change", False),
                include_last_updated_at=function_args.get("include_last_updated_at", False),
                precision=function_args.get("precision", None),
            )
            if "error" not in result:
                return {"price_data": result}
            return result

        elif tool_name == "get_categories_list":
            return await self._get_categories_list()

        elif tool_name == "get_category_data":
            order = function_args.get("order", "market_cap_desc")
            return await self._get_category_data(order)

        elif tool_name == "get_tokens_by_category":
            return await self._get_tokens_by_category(
                category_id=function_args["category_id"],
                vs_currency=function_args.get("vs_currency", "usd"),
                order=function_args.get("order", "market_cap_desc"),
                per_page=function_args.get("per_page", 100),
                page=function_args.get("page", 1),
            )

        elif tool_name == "get_trending_pools":
            return await self._handle_trending_pools(
                include=function_args.get("include", "base_token"),
                pools=function_args.get("pools", 4),
            )

        elif tool_name == "get_top_token_holders":
            return await self._handle_top_token_holders(
                network=function_args["network"],
                address=function_args["address"],
            )

        else:
            return {"error": f"Unsupported tool: {tool_name}"}

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def _handle_trending_pools(self, include: str, pools: int) -> Dict[str, Any]:
        """Handle trending pools API call"""
        valid_includes = ["base_token", "quote_token", "dex", "network"]
        if include not in valid_includes:
            logger.error(f"Invalid include parameter: {include}. Must be one of {valid_includes}")
            return {"error": f"Invalid include parameter: {include}. Must be one of {valid_includes}"}

        try:
            api_url, headers = self.get_api_config("/onchain/pools/trending_search")
            url = f"{api_url}/onchain/pools/trending_search"
            params = {"include": include, "pools": pools}
            response = await self._api_request(url=url, method="GET", headers=headers, params=params)
            if "error" in response:
                return {"error": response["error"]}
            return {"trending_pools": response}
        except Exception as e:
            logger.error(f"Error getting trending pools: {e}")
            return {"error": f"Failed to fetch trending pools: {str(e)}"}

    @with_cache(ttl_seconds=300)  # Cache for 5 minutes
    @with_retry(max_retries=3)
    async def _handle_top_token_holders(self, network: str, address: str) -> Dict[str, Any]:
        """Handle top token holders API call"""
        try:
            api_url, headers = self.get_api_config(f"/onchain/networks/{network}/tokens/{address}/top_holders")
            url = f"{api_url}/onchain/networks/{network}/tokens/{address}/top_holders"
            response = await self._api_request(url=url, method="GET", headers=headers)
            if "error" in response:
                return {"error": response["error"]}
            return {"top_holders": response}
        except Exception as e:
            logger.error(f"Error getting top token holders: {e}")
            return {"error": f"Failed to fetch top token holders: {str(e)}"}
