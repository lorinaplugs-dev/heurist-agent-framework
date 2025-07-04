import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class ArbusAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("ARBUS_API_KEY")
        if not self.api_key:
            raise ValueError("ARBUS_API_KEY environment variable is required")

        self.base_url = "https://api.arbus.ai/v1"
        self.headers = {"accept": "application/json"}

        self.metadata.update(
            {
                "name": "Arbus Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent provides cryptocurrency analysis, sentiment tracking, and market intelligence using the Arbus AI API.",
                "external_apis": ["Arbus"],
                "tags": ["Crypto Analysis"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Arbus.png",
                "examples": [
                    "Is Bitcoin bullish right now?",
                    "Get analysis for Solana project",
                    "Generate report for Ethereum projects",
                ],
                "credits": 0,
            }
        )

    def get_system_prompt(self) -> str:
        return """You are an expert cryptocurrency analyst with access to real-time market data through Arbus AI. When users ask about crypto markets, sentiment, or projects, I analyze the latest data and provide clear, objective insights.

                I deliver responses as direct analysis and findings, not as if I'm answering a user's question. I present my analysis confidently using phrases like "Current data shows..." or "Analysis indicates..." rather than "Here's what I found for you..."

                I focus on:
                - Current market sentiment with supporting data points
                - Key price drivers and market developments
                - Technical indicators and on-chain metrics
                - Project fundamentals and ecosystem updates
                - Risk factors and market dynamics

                I provide factual, data-driven analysis without speculation or investment advice. When data is limited, I clearly state the constraints while still offering relevant context from available information."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "ask_ai_assistant",
                    "description": "Query the Arbus AI assistant for real-time cryptocurrency market analysis, sentiment insights, and general market intelligence. This tool provides comprehensive answers to crypto-related questions using current market data, social sentiment, and on-chain metrics. Ideal for getting broad market overviews, understanding current trends, or asking specific questions about market conditions.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The specific question or topic about cryptocurrency markets you want analyzed. Examples: 'What is the current sentiment for Bitcoin?', 'How are DeFi tokens performing?', 'What are the main market drivers today?'",
                            },
                            "days": {
                                "type": "integer",
                                "description": "Number of days of historical data to include in the analysis (1-365 days). Default is 7 days. Use shorter periods (1-7 days) for recent events, longer periods (30-365 days) for trend analysis.",
                                "default": 7,
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "assistant_summary",
                    "description": "Generate a comprehensive AI-powered analysis summary for a specific cryptocurrency project or token. This tool provides detailed insights including bullish and bearish scenarios, key metrics, recent developments, and risk assessment. Perfect for deep-dive analysis of individual projects, understanding project fundamentals, and getting balanced perspectives on investment potential.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "ticker_or_twitterhandle": {
                                "type": "string",
                                "description": "The cryptocurrency ticker symbol (e.g., 'BTC', 'ETH', 'SOL') or the project's Twitter handle (with or without @). For Twitter handles, the @ symbol will be automatically removed if present.",
                            },
                            "day_interval": {
                                "type": "integer",
                                "description": "Number of days to analyze for the project summary (1-30 days). Default is 7 days. Use shorter intervals (1-7 days) for recent project updates, longer intervals (14-30 days) for comprehensive trend analysis.",
                                "default": 7,
                            },
                        },
                        "required": ["ticker_or_twitterhandle"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "report",
                    "description": "Generate structured, categorized reports for cryptocurrency projects based on their social media presence and community activity. This tool creates organized reports with specific focus areas like project updates, community sentiment, or alert-worthy events. Useful for tracking project development, monitoring community engagement, and generating formal analysis reports.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "twitter_handle": {
                                "type": "string",
                                "description": "The project's Twitter handle without the @ symbol (e.g., 'ethereum', 'solana', 'chainlink'). The @ symbol will be automatically removed if included.",
                            },
                            "category": {
                                "type": "string",
                                "description": "The type of report to generate: 'projects' for general project analysis and updates, 'threador' for Twitter thread analysis and social sentiment, 'alerts' for notable events and important announcements.",
                                "enum": ["projects", "threador", "alerts"],
                            },
                            "date_from": {
                                "type": "string",
                                "description": "Start date for the report analysis in YYYY-MM-DD format (e.g., '2024-01-01'). If not provided, the report will use a default recent time period.",
                            },
                            "date_to": {
                                "type": "string",
                                "description": "End date for the report analysis in YYYY-MM-DD format (e.g., '2024-01-31'). If not provided, the report will use the current date as the end point.",
                            },
                        },
                        "required": ["twitter_handle", "category"],
                    },
                },
            },
        ]

    # ---------------------
    # ARBUS API-SPECIFIC METHODS
    # ---------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def ask_ai_assistant(self, query: str, days: int = 7) -> Dict:
        logger.info(f"Fetching AI assistant response for query: {query}, days: {days}")
        url = f"{self.base_url}/ask-ai-assistant"
        params = {"key": self.api_key}
        json_data = {"query": query, "days": min(max(days, 1), 365)}
        return await self._api_request(url, method="POST", headers=self.headers, params=params, json_data=json_data)

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def assistant_summary(self, ticker_or_twitterhandle: str, day_interval: int) -> Dict:
        logger.info(f"Fetching assistant summary for: {ticker_or_twitterhandle}, day_interval: {day_interval}")
        url = f"{self.base_url}/assistant-summary"
        params = {"key": self.api_key}
        json_data = {
            "ticker_or_twitterhandle": ticker_or_twitterhandle.replace("@", ""),
            "day_interval": min(max(day_interval, 1), 30),
        }
        return await self._api_request(url, method="POST", headers=self.headers, params=params, json_data=json_data)

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def report(self, twitter_handle: str, category: str, date_from: str = None, date_to: str = None) -> Dict:
        logger.info(f"Fetching report for twitter_handle: {twitter_handle}, category: {category}")
        url = f"{self.base_url}/report"
        params = {"key": self.api_key}
        json_data = {
            "twitter_handle": twitter_handle.replace("@", ""),
            "category": category,
        }
        if date_from:
            json_data["date_from"] = date_from
        if date_to:
            json_data["date_to"] = date_to
        return await self._api_request(url, method="POST", headers=self.headers, params=params, json_data=json_data)

    # ---------------------
    # TOOL HANDLING LOGIC
    # ---------------------
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        logger.info(f"Handling tool: {tool_name} with args: {function_args}")
        if tool_name == "ask_ai_assistant":
            query = function_args.get("query")
            days = function_args.get("days", 7)
            if not query:
                logger.error("Missing 'query' in tool arguments")
                return {"error": "Missing 'query' in tool arguments"}
            result = await self.ask_ai_assistant(query, days)

        elif tool_name == "assistant_summary":
            ticker_or_twitterhandle = function_args.get("ticker_or_twitterhandle")
            day_interval = function_args.get("day_interval", 7)
            if not ticker_or_twitterhandle:
                logger.error("Missing 'ticker_or_twitterhandle' in tool arguments")
                return {"error": "Missing 'ticker_or_twitterhandle' in tool arguments"}
            result = await self.assistant_summary(ticker_or_twitterhandle, day_interval)

        elif tool_name == "report":
            twitter_handle = function_args.get("twitter_handle")
            category = function_args.get("category")
            date_from = function_args.get("date_from")
            date_to = function_args.get("date_to")
            if not twitter_handle or not category:
                logger.error("Missing 'twitter_handle' or 'category' in tool arguments")
                return {"error": "Missing 'twitter_handle' or 'category' in tool arguments"}
            result = await self.report(twitter_handle, category, date_from, date_to)

        else:
            logger.error(f"Unsupported tool: {tool_name}")
            return {"error": f"Unsupported tool '{tool_name}'"}

        if errors := self._handle_error(result):
            logger.warning(f"Error in tool execution: {errors}")
            return errors

        return {"data": result}
