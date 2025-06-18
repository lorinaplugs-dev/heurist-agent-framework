import asyncio
import logging
import os
import time
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
        return """You are a blockchain data processor that extracts and formats relevant information from blockchain explorer pages.

        Extract only the essential information and format it cleanly using the following structure:

        For TRANSACTIONS:
        - **Transaction Hash**: [hash value]
        - **Status**: [Success/Failed/Pending]
        - **Block Number**: [number] | **Timestamp**: [date/time]
        - **From**: [0xaddress](explorer_url/address/0xaddress)
        - **To**: [0xaddress](explorer_url/address/0xaddress)
        - **Value**: [amount] ETH/token
        - **Transaction Fee**: [fee amount]
        - **Gas Details**: Used [used]/[limit] | Price: [price] Gwei

        For ADDRESSES:
        - **Address**: [0xaddress]
        - **Balance**: [amount] ETH/tokens
        - **Transaction Count**: [number]

        **Recent Transactions** (if available):
        | Hash | Method | Age | From/To | Value |
        |------|--------|-----|---------|-------|
        | [hash] | [method] | [time] | [address] | [amount] |

        For TOKENS:
        - **Token Name**: [name] ([symbol])
        - **Contract Address**: [0xaddress](explorer_url/token/0xaddress)
        - **Total Supply**: [amount] [symbol]
        - **Holders**: [number] addresses
        - **Decimals**: [number]

        **Recent Transfers** (if available):
        | Txn Hash | Age | From | To | Value |
        |----------|-----|------|----|----|
        | [hash] | [time] | [from] | [to] | [amount] |

        IMPORTANT FORMATTING RULES:
        - Remove all advertisements, navigation menus, headers, footers, and irrelevant content
        - Format all addresses as clickable markdown links: [0xaddress](full_explorer_url)
        - Use tables for transaction/transfer lists with proper markdown syntax
        - Handle missing data gracefully by showing "N/A" or "Not available"
        - Keep monetary values with appropriate units (ETH, USD, etc.)
        - Present timestamps in human-readable format
        - Use bold formatting for field labels"""

    async def _process_with_llm(self, raw_content: str, context_info: Dict[str, str]) -> str:
        """Process raw scraped content with LLM and track performance"""
        start_time = time.time()

        try:
            messages = [
                {"role": "system", "content": self.get_system_prompt()},
                {
                    "role": "user",
                    "content": f"""Extract and format the relevant blockchain data from this {context_info.get("type", "page")} content.

                Context:
                - Chain: {context_info.get("chain", "unknown")}
                - Explorer URL: {context_info.get("url", "unknown")}
                - Data Type: {context_info.get("type", "unknown")}

                Raw Content:
                {raw_content}""",
                },
            ]

            if hasattr(self, "llm_client") and hasattr(self.llm_client, "chat"):
                # If using a direct LLM client
                response = await self.llm_client.chat(
                    model=self.metadata["small_model_id"], messages=messages, max_tokens=2000, temperature=0.1
                )
                processed_content = response.content if hasattr(response, "content") else str(response)
            elif hasattr(self, "generate_response"):
                # If using generate_response method
                processed_content = await self.generate_response(
                    messages=messages, model_id=self.metadata["small_model_id"], max_tokens=2000, temperature=0.1
                )
            elif hasattr(self, "call_llm"):
                # If call_llm exists
                processed_content = await self.call_llm(
                    messages=messages, model_id=self.metadata["small_model_id"], max_tokens=2000, temperature=0.1
                )
            else:
                prompt = f"{messages[0]['content']}\n\nUser: {messages[1]['content']}"
                if hasattr(self, "process_with_ai"):
                    processed_content = await self.process_with_ai(prompt)
                else:
                    logger.warning("No LLM method found in MeshAgent base class, returning raw content")
                    return raw_content

            processing_time = time.time() - start_time
            logger.info(f"LLM processing completed in {processing_time:.2f}s for {context_info.get('type', 'content')}")

            return processed_content

        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"LLM processing failed after {processing_time:.2f}s: {str(e)}")
            logger.warning("Falling back to raw content due to LLM processing failure")
            return raw_content

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
                None, lambda: self.app.scrape_url(url, formats=["markdown"], wait_for=5000)
            )

            markdown_content = getattr(scrape_result, "markdown", None) or (
                scrape_result.get("markdown") if isinstance(scrape_result, dict) else None
            )

            if not scrape_result or not markdown_content:
                return {"status": "error", "error": "Failed to scrape transaction page"}

            context_info = {"type": "transaction", "chain": chain, "url": url}

            processed_content = await self._process_with_llm(markdown_content, context_info)
            logger.info("Successfully processed transaction data")

            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "txid": txid,
                    "explorer_url": url,
                    "processed_content": processed_content,
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
                None, lambda: self.app.scrape_url(url, formats=["markdown"], wait_for=5000)
            )

            markdown_content = getattr(scrape_result, "markdown", None) or (
                scrape_result.get("markdown") if isinstance(scrape_result, dict) else None
            )

            if not scrape_result or not markdown_content:
                return {"status": "error", "error": "Failed to scrape address page"}

            context_info = {"type": "address", "chain": chain, "url": url}

            processed_content = await self._process_with_llm(markdown_content, context_info)
            logger.info("Successfully processed address data")

            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "address": address,
                    "explorer_url": url,
                    "processed_content": processed_content,
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
                None, lambda: self.app.scrape_url(url, formats=["markdown"], wait_for=5000)
            )

            markdown_content = getattr(scrape_result, "markdown", None) or (
                scrape_result.get("markdown") if isinstance(scrape_result, dict) else None
            )

            if not scrape_result or not markdown_content:
                return {"status": "error", "error": "Failed to scrape token page"}

            context_info = {"type": "token", "chain": chain, "url": url}

            processed_content = await self._process_with_llm(markdown_content, context_info)
            logger.info("Successfully processed token data")

            return {
                "status": "success",
                "data": {
                    "chain": chain,
                    "token_address": address,
                    "explorer_url": url,
                    "processed_content": processed_content,
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
        """Handle execution of specific tools and return the processed data"""

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
