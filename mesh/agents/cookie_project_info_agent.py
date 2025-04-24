import logging
import os
import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class CookieProjectInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.base_url = "https://api.staging.cookie.fun"
        self.api_key = os.getenv("COOKIE_FUN_API_KEY")
        if not self.api_key:
            raise ValueError("COOKIE_FUN_API_KEY environment variable is required")

        self.timeout = 10
        logger.info(f"Base URL set to {self.base_url}, timeout={self.timeout}")

        self.metadata.update(
            {
                "name": "Cookie Project Info Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent provides information about crypto projects using Cookie API, including project details, mindshare metrics, and trend analysis.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about crypto projects, their details, or mindshare.",
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
                        "description": "Structured data containing project details or mindshare metrics.",
                        "type": "dict",
                    },
                ],
                "external_apis": ["Cookie API"],
                "tags": ["Projects"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/CookieFun.png",
                "examples": [
                    "Tell me about the Heurist project",
                    "Search for DeFi projects",
                    "What is the mindshare trend for Ethereum?",
                    "Show me the top projects by mindshare in the AI sector",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """
        You are a crypto project information specialist that provides insights on blockchain projects using Cookie API data.

        Focus on delivering factual information and insights that help users understand:
        - Project details including name, symbol, contracts, and price metrics
        - Mindshare trends showing project popularity over time
        - Which projects are trending or gaining attention
        - Project rankings by various metrics

        If a user asks about a specific project but doesn't provide a slug, use the search_projects tool first to find the correct slug, then use that to get detailed information.

        Keep your responses concise and data-driven while making the information accessible. NEVER make up data that is not returned from the tool.
        """

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_project_details",
                    "description": "Get detailed information about a specific crypto project by slug or contract address.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "slug": {
                                "type": "string",
                                "description": "Project identifier/slug (e.g., 'heurist', 'bitcoin')",
                            },
                            "contract_address": {
                                "type": "string",
                                "description": "Token contract address",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_projects",
                    "description": "Search for crypto projects by name, symbol, or other keywords.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "search_query": {
                                "type": "string",
                                "description": "Search term (e.g., 'DeFi', 'Ethereum', 'AI')",
                            },
                            "page": {
                                "type": "integer",
                                "description": "Page number for pagination",
                                "default": 1,
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results to return (max 100)",
                                "default": 20,
                            },
                        },
                        "required": ["search_query"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_mindshare_graph",
                    "description": "Get mindshare trend data for a specific project over the past week.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "project_slug": {
                                "type": "string",
                                "description": "Project identifier/slug (e.g., 'heurist', 'bitcoin')",
                            },
                        },
                        "required": ["project_slug"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_mindshare_leaderboard",
                    "description": "Retrieves a list of crypto projects ranked by their mindshare metrics. Use this tool to find trending projects. Sort by mindshare to find top projects, or sort by mindshare delta to find fast-growing projects that recently started to gain attention. Available sectors are: layer-2, zero-knowledge, ai, layer-1, depin, gaming, meme, infrastructure, defi",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "timeframe": {
                                "type": "integer",
                                "description": "Timeframe for mindshare metrics (1=24h, 2=7d, 3=30d)",
                                "default": 2,
                                "enum": [1, 2, 3],
                            },
                            "sector_slug": {
                                "type": "string",
                                "description": "Filter by sector (e.g., 'defi', 'ai', 'layer-1')",
                                "enum": [
                                    "layer-2",
                                    "zero-knowledge",
                                    "ai",
                                    "layer-1",
                                    "depin",
                                    "gaming",
                                    "meme",
                                    "infrastructure",
                                    "defi",
                                ],
                            },
                            "sort_by": {
                                "type": "string",
                                "description": "Sort field (default is mindshare)",
                                "enum": ["mindshare", "mindshareDelta"],
                                "default": "mindshare",
                            },
                        },
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                       SHARED / UTILITY METHODS
    # ------------------------------------------------------------------------
    def _headers(self) -> dict:
        """Get headers for API requests with authentication"""
        return {"x-api-key": self.api_key, "Content-Type": "application/json"}

    def _extract_project_slug_from_query(self, query: str) -> Optional[str]:
        """
        Extract a potential project slug from a query
        """
        if not query:
            return None

        # Common patterns for project mentions
        project_patterns = [
            r"about\s+([a-zA-Z0-9_-]+)(?:\s+project)?",
            r"([a-zA-Z0-9_-]+)\s+project",
            r"project\s+([a-zA-Z0-9_-]+)",
            r"details\s+(?:for|of|about)\s+([a-zA-Z0-9_-]+)",
            r"information\s+(?:for|of|about)\s+([a-zA-Z0-9_-]+)",
            r"mindshare\s+(?:for|of)\s+([a-zA-Z0-9_-]+)",
        ]

        for pattern in project_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                potential_slug = match.group(1).lower()

                # Filter out common words that might be incorrectly matched
                common_words = [
                    "the",
                    "project",
                    "token",
                    "crypto",
                    "blockchain",
                    "details",
                    "information",
                    "statistics",
                    "metrics",
                    "data",
                    "show",
                    "get",
                    "find",
                    "tell",
                    "me",
                ]
                if potential_slug.lower() not in common_words:
                    return potential_slug

        return None

    # ------------------------------------------------------------------------
    #                      COOKIE API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_project_details(self, slug: str = None, contract_address: str = None) -> Dict:
        """
        Get detailed information about a specific crypto project

        Args:
            slug: Project slug/identifier
            contract_address: Token contract address

        Returns:
            Dict with project details or error information
        """
        tool = "get_project_details"
        logger.info(f"[Tool Start] {tool} - slug={slug}, contract_address={contract_address}")

        if not slug and not contract_address:
            return {"error": "Provide either slug or contract_address."}

        payload = {}
        if slug:
            payload["slug"] = slug
        if contract_address:
            payload["contractAddress"] = contract_address

        try:
            url = f"{self.base_url}/v3/project"
            response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)

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
    async def search_projects(self, search_query: str, page: int = 1, limit: int = 20) -> Dict:
        """
        Search for crypto projects by keywords

        Args:
            search_query: Search term
            page: Page number for pagination
            limit: Maximum number of results to return

        Returns:
            Dict with search results or error information
        """
        tool = "search_projects"
        logger.info(f"[Tool Start] {tool} - query={search_query}, page={page}, limit={limit}")

        try:
            url = f"{self.base_url}/v3/project/search"
            payload = {"searchQuery": search_query, "page": page, "limit": min(limit, 100)}

            response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)

            if response.status_code == 401:
                logger.error(f"[Error] {tool} returned 401 Unauthorized")
                return {"error": "Authorization failed. Check your API key."}

            response.raise_for_status()
            result = response.json()
            logger.info(f"[Tool End] {tool} - Successfully retrieved search results")
            return {"status": "success", "data": result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error searching projects: {str(e)}")
            return {"error": f"Failed to search projects: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_mindshare_graph(self, project_slug: str) -> Dict:
        """
        Get mindshare trend data for a specific project

        Args:
            project_slug: Project slug/identifier

        Returns:
            Dict with mindshare trend data or error information
        """
        tool = "get_mindshare_graph"
        logger.info(f"[Tool Start] {tool} - project_slug={project_slug}")

        try:
            url = f"{self.base_url}/v3/project/mindshare-graph"

            end = datetime.utcnow()
            start = end - timedelta(days=7)
            fmt = "%Y-%m-%dT%H:%M"

            payload = {
                "projectSlug": project_slug,
                "startDate": start.strftime(fmt),
                "endDate": end.strftime(fmt),
                "granulation": 1,
            }

            response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)

            if response.status_code == 401:
                logger.error(f"[Error] {tool} returned 401 Unauthorized")
                return {"error": "Authorization failed. Check your API key."}

            response.raise_for_status()
            result = response.json()
            logger.info(f"[Tool End] {tool} - Successfully retrieved mindshare graph")
            return {"status": "success", "data": result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching mindshare graph: {str(e)}")
            return {"error": f"Failed to fetch mindshare graph: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_mindshare_leaderboard(self, timeframe: int = 2, sector_slug: str = None, sort_by: str = None) -> Dict:
        """
        Get mindshare leaderboard data

        Args:
            timeframe: Timeframe for data (1=24h, 2=7d, 3=30d)
            sector_slug: Filter by sector
            sort_by: Field to sort by

        Returns:
            Dict with mindshare leaderboard data or error information
        """
        tool = "get_mindshare_leaderboard"
        logger.info(f"[Tool Start] {tool} - timeframe={timeframe}, sector={sector_slug}, sort_by={sort_by}")

        try:
            url = f"{self.base_url}/v3/project/mindshare-leaderboard"

            payload = {"mindshareTimeframe": timeframe, "sortOrder": 1}
            if sector_slug:
                payload["sectorSlug"] = sector_slug
            if sort_by:
                payload["sortBy"] = sort_by

            response = requests.post(url, headers=self._headers(), json=payload, timeout=self.timeout)

            if response.status_code == 401:
                logger.error(f"[Error] {tool} returned 401 Unauthorized")
                return {"error": "Authorization failed. Check your API key."}

            response.raise_for_status()
            result = response.json()
            logger.info(f"[Tool End] {tool} - Successfully retrieved mindshare leaderboard")
            return {"status": "success", "data": result}

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching mindshare leaderboard: {str(e)}")
            return {"error": f"Failed to fetch mindshare leaderboard: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """Handle tool execution and return results."""
        tool_functions = {
            "get_project_details": self.get_project_details,
            "search_projects": self.search_projects,
            "get_mindshare_graph": self.get_mindshare_graph,
            "get_mindshare_leaderboard": self.get_mindshare_leaderboard,
        }

        if tool_name not in tool_functions:
            return {"error": f"Unsupported tool: {tool_name}"}

        if tool_name == "get_project_details":
            result = await tool_functions[tool_name](
                slug=function_args.get("slug"),
                contract_address=function_args.get("contract_address"),
            )

        elif tool_name == "search_projects":
            search_query = function_args.get("search_query")
            if not search_query:
                return {"error": "Missing 'search_query' in tool_arguments"}
            result = await tool_functions[tool_name](
                search_query=search_query,
                page=function_args.get("page", 1),
                limit=function_args.get("limit", 20),
            )

        elif tool_name == "get_mindshare_graph":
            project_slug = function_args.get("project_slug")
            if not project_slug:
                return {"error": "Missing 'project_slug' in tool_arguments"}
            result = await tool_functions[tool_name](project_slug=project_slug)

        elif tool_name == "get_mindshare_leaderboard":
            result = await tool_functions[tool_name](
                timeframe=function_args.get("timeframe", 2),
                sector_slug=function_args.get("sector_slug"),
                sort_by=function_args.get("sort_by"),
            )

        errors = self._handle_error(result)
        return errors if errors else result

    # ------------------------------------------------------------------------
    #                       MAIN MESSAGE HANDLER
    # ------------------------------------------------------------------------
    async def _before_handle_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Hook called before message handling to preprocess parameters.
        Handles extraction of project slugs from natural language queries.
        """
        query = params.get("query")

        if query and not params.get("tool"):
            potential_slug = self._extract_project_slug_from_query(query)

            if potential_slug:
                logger.info(f"Extracted potential project slug: '{potential_slug}' from query: '{query}'")
                thinking_msg = f"Looking up information for project: {potential_slug}..."
                self.push_update(params, thinking_msg)

                search_result = await self.search_projects(search_query=potential_slug, limit=1)

                if (
                    "status" in search_result
                    and search_result["status"] == "success"
                    and "data" in search_result
                    and "ok" in search_result["data"]
                    and "entries" in search_result["data"]["ok"]
                    and len(search_result["data"]["ok"]["entries"]) > 0
                ):
                    actual_slug = search_result["data"]["ok"]["entries"][0]["slug"]

                    modified_params = params.copy()
                    modified_params["tool"] = "get_project_details"
                    modified_params["tool_arguments"] = {"slug": actual_slug}

                    return modified_params
        return await super()._before_handle_message(params)
