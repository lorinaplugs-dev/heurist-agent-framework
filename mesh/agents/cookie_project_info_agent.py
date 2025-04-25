import logging
import os
from typing import Any, Dict, List

import requests
from dotenv import load_dotenv

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class CookieProjectInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.cookie.fun"
        self.api_key = os.getenv("COOKIE_FUN_API_KEY")
        if not self.api_key:
            raise ValueError("COOKIE_FUN_API_KEY environment variable is required")

        self.timeout = 10
        self.headers = {"x-api-key": self.api_key, "Content-Type": "application/json"}
        logger.info(f"Base URL set to {self.base_url}, timeout={self.timeout}")

        self.metadata.update(
            {
                "name": "Cookie Project Info Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent provides information about crypto projects using Cookie API, including project details by Twitter username and contract address.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about crypto projects, their details, or rankings.",
                        "type": "str",
                        "required": False,
                    },
                    {
                        "name": "raw_data_only",
                        "description": "If true, return only raw data without natural language response.",
                        "type": "bool",
                        "required": False,
                        "default": False,
                    },
                ],
                "outputs": [
                    {
                        "name": "response",
                        "description": "Natural language explanation of the project information.",
                        "type": "str",
                    },
                    {
                        "name": "data",
                        "description": "Structured data containing project details.",
                        "type": "dict",
                    },
                ],
                "external_apis": ["Cookie API"],
                "tags": ["Projects"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/CookieFun.png",
                "examples": [
                    "Tell me about the project with Twitter handle @heurist_ai for past 30 days",
                    "Tell me about the project with Twitter handle @cookiedotfun",
                    "Get details for the contract 0xc0041ef357b183448b235a8ea73ce4e4ec8c265f",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """
        You are a crypto project information specialist that provides insights on blockchain projects using Cookie API data.

        Focus on delivering factual information and insights that help users understand:
        - Project details including information from Twitter username
        - Project information from contract addresses

        If a user asks about a specific project, try to identify whether they've provided a Twitter username
        or contract address and use the appropriate tool and also if interval exceeds more than 7 days, make it 7 days for fetching

        Keep your responses concise and data-driven while making the information accessible. NEVER make up data that is not returned from the tool.
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_project_by_twitter_username",
                    "description": "Get information about a crypto project by its Twitter username.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "twitter_username": {
                                "type": "string",
                                "description": "Twitter username without the @ symbol",
                            },
                            "interval": {
                                "type": "string",
                                "description": "Time interval for the data (_3Days, _7Days)",
                                "enum": ["_3Days", "_7Days"],
                                "default": "_7Days",
                            },
                        },
                        "required": ["twitter_username"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_project_by_contract_address",
                    "description": "Get information about a crypto project by its contract address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "contract_address": {
                                "type": "string",
                                "description": "Token contract address",
                            },
                            "interval": {
                                "type": "string",
                                "description": "Time interval for the data (_3Days, _7Days)",
                                "enum": ["_3Days", "_7Days"],
                                "default": "_7Days",
                            },
                        },
                        "required": ["contract_address"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                      COOKIE API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_project_by_twitter_username(self, twitter_username: str, interval: str = "_7Days") -> Dict:
        """
        Get information about a crypto project by its Twitter username.

        Args:
            twitter_username: Twitter username (without @ symbol)
            interval: Time interval for data (_7Days, _30Days, _90Days)

        Returns:
            Dict with project details or error information
        """
        tool = "get_project_by_twitter_username"
        logger.info(f"[Tool Start] {tool} - twitter_username={twitter_username}, interval={interval}")

        try:
            url = f"{self.base_url}/v2/agents/twitterUsername/{twitter_username}"
            params = {"interval": interval}

            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)

            if response.status_code == 401:
                logger.error(f"[Error] {tool} returned 401 Unauthorized")
                return {"error": "Authorization failed. Check your API key."}

            response.raise_for_status()
            result = response.json()
            logger.info(f"[Tool End] {tool} - Successfully retrieved project details")
            return {"status": "success", "data": result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching project details: {str(e)}")
            return {"error": f"Failed to fetch project details: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_project_by_contract_address(self, contract_address: str, interval: str = "_7Days") -> Dict:
        """
        Get information about a crypto project by its contract address.

        Args:
            contract_address: Token contract address
            interval: Time interval for data (_7Days, _30Days, _90Days)

        Returns:
            Dict with project details or error information
        """
        tool = "get_project_by_contract_address"
        logger.info(f"[Tool Start] {tool} - contract_address={contract_address}, interval={interval}")

        try:
            url = f"{self.base_url}/v2/agents/contractAddress/{contract_address}"
            params = {"interval": interval}

            response = requests.get(url, headers=self.headers, params=params, timeout=self.timeout)

            if response.status_code == 401:
                logger.error(f"[Error] {tool} returned 401 Unauthorized")
                return {"error": "Authorization failed. Check your API key."}

            response.raise_for_status()
            result = response.json()
            logger.info(f"[Tool End] {tool} - Successfully retrieved project details")
            return {"status": "success", "data": result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching project details: {str(e)}")
            return {"error": f"Failed to fetch project details: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """Handle tool execution and return results."""
        tool_functions = {
            "get_project_by_twitter_username": self.get_project_by_twitter_username,
            "get_project_by_contract_address": self.get_project_by_contract_address,
        }

        if tool_name not in tool_functions:
            return {"error": f"Unsupported tool: {tool_name}"}

        if tool_name == "get_project_by_twitter_username":
            twitter_username = function_args.get("twitter_username")
            if not twitter_username:
                return {"error": "Missing 'twitter_username' in tool_arguments"}
            result = await tool_functions[tool_name](
                twitter_username=twitter_username,
                interval=function_args.get("interval", "_7Days"),
            )

        elif tool_name == "get_project_by_contract_address":
            contract_address = function_args.get("contract_address")
            if not contract_address:
                return {"error": "Missing 'contract_address' in tool_arguments"}
            result = await tool_functions[tool_name](
                contract_address=contract_address,
                interval=function_args.get("interval", "_7Days"),
            )

        errors = self._handle_error(result)
        return errors if errors else result
