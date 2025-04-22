import logging
import os
from typing import Any, Dict, List

import aiohttp
from dotenv import load_dotenv

from decorators import monitor_execution, with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class UnifaiTokenAnalysisAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("UNIFAI_API_KEY")
        if not self.api_key:
            raise ValueError("UNIFAI_API_KEY environment variable is required")

        self.api_endpoint = "https://backend.unifai.network/api/v1/actions/call"
        self.gmgn_trend_id = 39
        self.token_analysis_id = 25

        self.metadata.update(
            {
                "name": "UnifAI Token Analysis Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent provides token analysis using UnifAI's API, including GMGN trend analysis (GMGN is a memecoin trading platform) and comprehensive token analysis for various cryptocurrencies",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query for token analysis or trending tokens",
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
                        "description": "Natural language summary of token analysis",
                        "type": "str",
                    },
                    {
                        "name": "data",
                        "description": "Structured data containing token analysis results",
                        "type": "dict",
                    },
                ],
                "external_apis": ["UnifAI"],
                "tags": ["Token Analysis"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Unifai.png",
                "examples": [
                    "Show me trending tokens on GMGN",
                    "Analyze the ETH token for me",
                    "What are the top 10 trending tokens on GMGN in the last 24 hours?",
                    "Get token information for 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 on Ethereum",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """
        You are a token analysis specialist that helps users understand cryptocurrency tokens and market trends. You provide clear market insights and metrics. You highlight key data points for investors. Be factual and objective in your analysis. NEVER make up any data not returned by the tools
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_gmgn_trend",
                    "description": "Get trending tokens from GMGN memecoin trading platform",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "time_window": {
                                "type": "string",
                                "description": "Time window for trends (e.g., '24h', '4h', '1h')",
                                "default": "24h",
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of trending tokens to return",
                                "default": 50,
                                "minimum": 1,
                                "maximum": 100,
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_gmgn_token_info",
                    "description": "Get detailed token information from GMGN",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Blockchain network (e.g., 'eth', 'sol', 'base', 'bsc')",
                            },
                            "address": {
                                "type": "string",
                                "description": "Token contract address",
                            },
                        },
                        "required": ["chain", "address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_token",
                    "description": "Analyze a cryptocurrency token by its ticker",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker": {
                                "type": "string",
                                "description": "Token ticker symbol (e.g., 'ETH', 'BTC', 'SOL')",
                            },
                        },
                        "required": ["ticker"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                       API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_gmgn_trend(self, time_window: str = "24h", limit: int = 50) -> Dict:
        """
        Fetches trending tokens from GMGN memecoin trading platform

        Args:
            time_window: Time window for trends (e.g., '24h', '4h', '1h')
            limit: Maximum number of trending tokens to return

        Returns:
            Dict containing trending tokens or error information
        """
        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        try:
            action = f"GMGNTrendTokenAnalysis/{self.gmgn_trend_id}/gmgntrend"
            payload = {"timeWindow": time_window, "limit": limit}

            logger.info(f"Fetching GMGN trends for {time_window} with limit {limit}")

            async with aiohttp.ClientSession() as session:
                data = {"action": action, "payload": payload}
                async with session.post(self.api_endpoint, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully fetched GMGN trends")
                        return {"status": "success", "data": result}
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return {"status": "error", "error": f"API error: {response.status} - {error_text}"}

        except Exception as e:
            logger.error(f"Error fetching GMGN trends: {str(e)}")
            return {"status": "error", "error": f"Failed to get GMGN trends: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=600)
    @with_retry(max_retries=3)
    async def get_gmgn_token_info(self, chain: str, address: str) -> Dict:
        """
        Fetches specific token information from GMGN

        Args:
            chain: Blockchain network (e.g., 'eth', 'sol', 'base', 'bsc')
            address: Token contract address

        Returns:
            Dict containing token information or error details
        """
        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        try:
            action = f"GMGNTrendTokenAnalysis/{self.gmgn_trend_id}/getgmgntokeninfo"
            payload = {"chain": chain, "address": address}

            logger.info(f"Fetching GMGN token info for {address} on {chain}")

            async with aiohttp.ClientSession() as session:
                data = {"action": action, "payload": payload}
                async with session.post(self.api_endpoint, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info("Successfully fetched GMGN token info")
                        return {"status": "success", "data": result}
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return {"status": "error", "error": f"API error: {response.status} - {error_text}"}

        except Exception as e:
            logger.error(f"Error fetching GMGN token info: {str(e)}")
            return {"status": "error", "error": f"Failed to get GMGN token info: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def analyze_token(self, ticker: str) -> Dict:
        """
        Analyzes a token by its ticker symbol

        Args:
            ticker: Token ticker symbol (e.g., 'ETH', 'BTC', 'SOL')

        Returns:
            Dict containing token analysis or error information
        """
        headers = {"Authorization": self.api_key, "Content-Type": "application/json"}
        try:
            action = f"TokenAnalysis/{self.token_analysis_id}/analyzeToken"
            payload = {"ticker": ticker}

            logger.info(f"Analyzing token: {ticker}")

            async with aiohttp.ClientSession() as session:
                data = {"action": action, "payload": payload}
                async with session.post(self.api_endpoint, headers=headers, json=data, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        logger.info(f"Successfully analyzed token: {ticker}")
                        return {"status": "success", "data": result}
                    else:
                        error_text = await response.text()
                        logger.error(f"API error: {response.status} - {error_text}")
                        return {"status": "error", "error": f"API error: {response.status} - {error_text}"}

        except Exception as e:
            logger.error(f"Error analyzing token {ticker}: {str(e)}")
            return {"status": "error", "error": f"Failed to analyze token: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """Handle tool execution logic"""
        if tool_name == "get_gmgn_trend":
            time_window = function_args.get("time_window", "24h")
            limit = function_args.get("limit", 50)

            logger.info(f"Getting GMGN trends for {time_window} with limit {limit}")
            result = await self.get_gmgn_trend(time_window=time_window, limit=limit)

            errors = self._handle_error(result)
            if errors:
                return errors

            return result

        elif tool_name == "get_gmgn_token_info":
            chain = function_args.get("chain")
            address = function_args.get("address")

            if not chain or not address:
                return {"error": "Both 'chain' and 'address' are required"}

            logger.info(f"Getting GMGN token info for {address} on {chain}")
            result = await self.get_gmgn_token_info(chain=chain, address=address)

            errors = self._handle_error(result)
            if errors:
                return errors

            return result

        elif tool_name == "analyze_token":
            ticker = function_args.get("ticker")

            if not ticker:
                return {"error": "Ticker is required"}

            logger.info(f"Analyzing token: {ticker}")
            result = await self.analyze_token(ticker=ticker)

            errors = self._handle_error(result)
            if errors:
                return errors

            return result

        else:
            return {"error": f"Unsupported tool: {tool_name}"}
