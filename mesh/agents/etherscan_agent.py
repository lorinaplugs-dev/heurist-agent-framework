import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from firecrawl import FirecrawlApp

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class EtherscanAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("FIRECRAWL_API_KEY")
        if not self.api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable is required")

        self.app = FirecrawlApp(api_key=self.api_key)

        # Supported blockchain
        self.explorers = {
            "ethereum": "https://etherscan.io",
            "base": "https://basescan.org",
            "arbitrum": "https://arbiscan.io",
            "zksync": "https://era.zksync.network",
            "avalanche": "https://snowscan.xyz",
            "bsc": "https://bscscan.com",
        }

        self.metadata.update(
            {
                "name": "Etherscan Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent can analyze blockchain transactions, addresses, and ERC20 tokens across multiple chains using blockchain explorers and Firecrawl for data extraction.",
                "external_apis": ["Firecrawl"],
                "tags": ["Blockchain"],
                "recommended": True,
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/Etherscan.png",
                "examples": [
                    "Analyze transaction 0xd8a484a402a4373221288fed84e9025ed48eba2a45a7294c19289f740ca00fcd on Ethereum",
                    "Get address history for 0x742d35Cc6639C0532fEa3BcdE3524A0be79C3b7B on Base",
                    "Show token details for 0x55d398326f99059ff775485246999027b3197955 on BSC",
                ],
                "credits": 2,
                "large_model_id": "google/gemini-2.0-flash-001",
                "small_model_id": "google/gemini-2.0-flash-001",
            }
        )

    def get_system_prompt(self) -> str:
        return """You are a web data parser designed to scrape data from blockchain explorers such as Etherscan, Basescan, and similar platforms.

        Your task:
        - Extract all relevant data from blockchain explorer pages, focusing on core sections like 'Overview', 'More Info', and transaction lists.
        - Capture every key field presented in these sections for transactions, addresses, and tokens (e.g., status, participants, amounts, gas fees, balances, token details, and any other fields present).
        - For transaction lists, present each entry row by row in a clear table format.
        - Exclude irrelevant content such as advertisements, headers, footers, and navigation links.
        - Preserve all factual data exactly as presented without adding interpretations, summaries, or assessments.
        - Format addresses as clickable links in the format: [0xaddress](explorer_url).
        - Return the extracted data in a structured, concise format optimized for clarity and token efficiency.

        Stay factual and focus on presenting the raw, relevant data directly from the scraped content."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_transaction_details",
                    "description": "Analyze a specific blockchain transaction by scraping the explorer page. Provides detailed information about the transaction including sender, receiver, amount, gas fees, and status.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Blockchain network to query",
                                "enum": ["ethereum", "base", "arbitrum", "zksync", "avalanche", "bsc"],
                            },
                            "txid": {
                                "type": "string",
                                "description": "Transaction hash to analyze",
                            },
                        },
                        "required": ["chain", "txid"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_address_history",
                    "description": "Analyze a blockchain address to get transaction history, balance, and activity summary. Provides insights into the address's usage patterns and holdings.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Blockchain network to query",
                                "enum": ["ethereum", "base", "arbitrum", "zksync", "avalanche", "bsc"],
                            },
                            "address": {
                                "type": "string",
                                "description": "Wallet address to analyze",
                            },
                        },
                        "required": ["chain", "address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_erc20_token_details",
                    "description": "Analyze an ERC20 token contract to get token information including name, symbol, supply, holders, and contract details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Blockchain network where the token is deployed",
                                "enum": ["ethereum", "base", "arbitrum", "zksync", "avalanche", "bsc"],
                            },
                            "address": {
                                "type": "string",
                                "description": "Token contract address to analyze",
                            },
                        },
                        "required": ["chain", "address"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                      ETHERSCAN API-SPECIFIC METHODS
    # ------------------------------------------------------------------------

    # using async executor to run firecrawl operation
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_transaction_details(self, chain: str, txid: str) -> Dict[str, Any]:
        """
        Scrape and analyze transaction details from blockchain explorer.
        """
        logger.info(f"Getting transaction details for {txid} on {chain}")

        try:
            if chain not in self.explorers:
                return {"status": "error", "error": f"Unsupported chain: {chain}"}

            explorer_url = self.explorers[chain]
            url = f"{explorer_url}/tx/{txid}"

            scrape_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.app.scrape_url(url, params={"formats": ["markdown"], "waitFor": 5000})
            )

            if not scrape_result or "markdown" not in scrape_result:
                return {"status": "error", "error": "Failed to scrape transaction page"}

            logger.info("Successfully scraped transaction data")
            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "txid": txid,
                    "explorer_url": url,
                    "scraped_content": scrape_result["markdown"],
                },
            }

        except Exception as e:
            logger.error(f"Exception in get_transaction_details: {str(e)}")
            return {"status": "error", "error": f"Failed to get transaction details: {str(e)}"}

    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def get_address_history(self, chain: str, address: str) -> Dict[str, Any]:
        """
        Scrape and analyze address history from blockchain explorer.
        """
        logger.info(f"Getting address history for {address} on {chain}")

        try:
            if chain not in self.explorers:
                return {"status": "error", "error": f"Unsupported chain: {chain}"}

            explorer_url = self.explorers[chain]
            url = f"{explorer_url}/address/{address}"

            scrape_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.app.scrape_url(url, params={"formats": ["markdown"], "waitFor": 5000})
            )

            if not scrape_result or "markdown" not in scrape_result:
                return {"status": "error", "error": "Failed to scrape address page"}

            logger.info("Successfully scraped address data")
            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "address": address,
                    "explorer_url": url,
                    "scraped_content": scrape_result["markdown"],
                },
            }

        except Exception as e:
            logger.error(f"Exception in get_address_history: {str(e)}")
            return {"status": "error", "error": f"Failed to get address history: {str(e)}"}

    @with_cache(ttl_seconds=3600)
    @with_retry(max_retries=3)
    async def get_erc20_token_details(self, chain: str, address: str) -> Dict[str, Any]:
        """
        Scrape and analyze ERC20 token details from blockchain explorer.
        """
        logger.info(f"Getting ERC20 token details for {address} on {chain}")

        try:
            if chain not in self.explorers:
                return {"status": "error", "error": f"Unsupported chain: {chain}"}

            explorer_url = self.explorers[chain]
            url = f"{explorer_url}/token/{address}"

            scrape_result = await asyncio.get_event_loop().run_in_executor(
                None, lambda: self.app.scrape_url(url, params={"formats": ["markdown"], "waitFor": 5000})
            )

            if not scrape_result or "markdown" not in scrape_result:
                return {"status": "error", "error": "Failed to scrape token page"}

            logger.info("Successfully scraped token data")
            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "token_address": address,
                    "explorer_url": url,
                    "scraped_content": scrape_result["markdown"],
                },
            }

        except Exception as e:
            logger.error(f"Exception in get_erc20_token_details: {str(e)}")
            return {"status": "error", "error": f"Failed to get token details: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Handle execution of specific tools and return the raw data"""

        logger.info(f"Handling tool call: {tool_name} with args: {function_args}")

        if tool_name == "get_transaction_details":
            chain = function_args.get("chain")
            txid = function_args.get("txid")

            if not chain or not txid:
                return {"status": "error", "error": "Missing required parameters: chain and txid"}

            result = await self.get_transaction_details(chain, txid)

        elif tool_name == "get_address_history":
            chain = function_args.get("chain")
            address = function_args.get("address")

            if not chain or not address:
                return {"status": "error", "error": "Missing required parameters: chain and address"}

            result = await self.get_address_history(chain, address)

        elif tool_name == "get_erc20_token_details":
            chain = function_args.get("chain")
            address = function_args.get("address")

            if not chain or not address:
                return {"status": "error", "error": "Missing required parameters: chain and address"}

            result = await self.get_erc20_token_details(chain, address)

        else:
            return {"status": "error", "error": f"Unsupported tool: {tool_name}"}

        errors = self._handle_error(result)
        if errors:
            return errors

        return result
