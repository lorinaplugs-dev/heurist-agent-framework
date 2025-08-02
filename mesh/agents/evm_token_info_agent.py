import logging
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from decorators import monitor_execution, with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class EvmTokenInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("BITQUERY_API_KEY")
        if not self.api_key:
            raise ValueError("BITQUERY_API_KEY environment variable is required")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.bitquery_url = "https://streaming.bitquery.io/graphql"

        self.metadata.update(
            {
                "name": "EVM Token Info Agent",
                "version": "1.2.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent analyzes large trades for EVM tokens across multiple chains using Bitquery API. It tracks whale movements, identifying large buyers and sellers with transaction details.",
                "external_apis": ["Bitquery"],
                "tags": ["EVM"],
                "supported_chains": ["ethereum", "eth", "bsc", "binance", "base", "arbitrum", "arb"],
                "examples": [
                    "Show me recent large trades for USDT on Ethereum",
                    "Get large buyers of token 0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c on BSC",
                    "Show recent whale sells for WETH on base with minimum $10,000",
                    "Large trades for 0xA0b86991c6218b36c1d19D4a2e9Eb0cE3606eB48 on ethereum above $50k",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """You are a specialized assistant that analyzes large trades for EVM tokens across multiple blockchains using Bitquery API. Your capabilities include:

        1. Large Trade Analysis: Track and analyze large trades (whales) for any EVM token on supported chains
        2. Buyer/Seller Identification: Identify large buyers and sellers with their addresses and transaction details
        3. Trade Filtering: Filter trades by minimum USD amount and trade type (buy/sell/all)
        4. Multi-chain Support: Analyze tokens on Ethereum (eth), Binance Smart Chain (bsc), Base (base), and Arbitrum (arbitrum)

        Key Features:
        - Default minimum trade size: $5,000 USD (customizable)
        - Shows both buyers and sellers by default unless specifically requested
        - Returns raw trade data from Bitquery API
        - Sorted by USD amount in descending order

        Important:
        - User must provide a valid token contract address (starting with 0x)
        - If no valid address is provided, inform the user to provide a valid EVM token contract address
        - Only use buy/sell filters when user explicitly asks for "buyers only", "sellers only", "only buyers", "only sellers", etc.
        - Present the data in a clear and concise manner focusing on key insights
        - Supported chains: Ethereum, BSC, Base, and Arbitrum only"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_recent_large_trades",
                    "description": "Get recent large trades for a specific EVM token on supported chains. Shows whale activity including large buys and sells with transaction details. Perfect for tracking smart money movements and identifying accumulation or distribution patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "chain": {
                                "type": "string",
                                "description": "Blockchain network. Supported: ethereum/eth, bsc/binance, base, arbitrum/arb",
                            },
                            "tokenAddress": {
                                "type": "string",
                                "description": "Token contract address (must start with 0x, e.g., '0x7130d2a12b9bcbfae4f2634d864a1ee1ce3ead9c')",
                            },
                            "minUsdAmount": {
                                "type": "number",
                                "description": "Minimum trade size in USD to filter whale trades",
                                "default": 5000,
                            },
                            "filter": {
                                "type": "string",
                                "description": "Filter by trade type: 'all' for both buys and sells (default), 'buy' for purchases only, 'sell' for sales only",
                                "enum": ["all", "buy", "sell"],
                                "default": "all",
                            },
                            "limit": {
                                "type": "number",
                                "description": "Number of trades to return",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                            },
                        },
                        "required": ["chain", "tokenAddress"],
                    },
                },
            }
        ]

    @monitor_execution()
    @with_cache(ttl_seconds=60)
    @with_retry(max_retries=3)
    async def get_recent_large_trades(
        self,
        chain: str,
        tokenAddress: str,
        minUsdAmount: float = 5000,
        filter: str = "all",
        limit: int = 10,
    ) -> Dict[str, Any]:
        """
        Get recent large trades for a specific EVM token.
        Returns raw data from Bitquery API without any processing.
        """
        if not tokenAddress or not tokenAddress.startswith("0x"):
            return {"error": "Please provide a valid token contract address starting with '0x'"}

        # Normalize chain names to Bitquery enum values
        chain_mapping = {
            "ethereum": "eth",
            "eth": "eth",
            "bsc": "bsc",
            "binance": "bsc",
            "bnb": "bsc",
            "base": "base",
            "arbitrum": "arbitrum",
            "arb": "arbitrum"
        }
        
        normalized_chain = chain_mapping.get(chain.lower())
        if not normalized_chain:
            return {"error": f"Unsupported chain: {chain}. Supported chains: ethereum/eth, bsc/binance, base, arbitrum/arb"}

        if filter not in ["all", "buy", "sell"]:
            return {"error": f"Invalid filter: {filter}. Must be 'all', 'buy', or 'sell'"}

        limit = min(max(1, limit), 100)

        # Build the query with the network as an enum (not string)
        if filter == "all":
            query = f"""query getRecentLargeTrades($token: String!, $minUsdAmount: String!, $limit: Int!) {{
  EVM(network: {normalized_chain}) {{
    DEXTradeByTokens(
      orderBy: {{descendingByField: "Trade_Side_AmountInUSD"}}
      limit: {{count: $limit}}
      where: {{
        Trade: {{
          Currency: {{SmartContract: {{is: $token}}}},
          Side: {{
            AmountInUSD: {{gt: $minUsdAmount}}
          }}
        }}
      }}
    ) {{
      Trade {{
        Buyer
        Seller
        Currency {{
          Name
          Symbol
          SmartContract
        }}
        Side {{
          Type
          Amount
          AmountInUSD
          Currency {{
            Name
            Symbol
            SmartContract
          }}
        }}
      }}
      Transaction {{
        Hash
      }}
      Block {{
        Time
      }}
    }}
  }}
}}"""
        else:
            query = f"""query getRecentLargeTrades($token: String!, $minUsdAmount: String!, $limit: Int!) {{
  EVM(network: {normalized_chain}) {{
    DEXTradeByTokens(
      orderBy: {{descendingByField: "Trade_Side_AmountInUSD"}}
      limit: {{count: $limit}}
      where: {{
        Trade: {{
          Currency: {{SmartContract: {{is: $token}}}},
          Side: {{
            Type: {{is: {filter}}},
            AmountInUSD: {{gt: $minUsdAmount}}
          }}
        }}
      }}
    ) {{
      Trade {{
        Buyer
        Seller
        Currency {{
          Name
          Symbol
          SmartContract
        }}
        Side {{
          Type
          Amount
          AmountInUSD
          Currency {{
            Name
            Symbol
            SmartContract
          }}
        }}
      }}
      Transaction {{
        Hash
      }}
      Block {{
        Time
      }}
    }}
  }}
}}"""

        variables = {
            "token": tokenAddress.lower(),
            "minUsdAmount": str(minUsdAmount),
            "limit": limit,
        }

        print("Generated query:", query)
        print("Variables:", variables)
        
        return await self._api_request(
            url=self.bitquery_url,
            method="POST",
            headers=self.headers,
            json_data={"query": query, "variables": variables},
        )

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle execution of specific tools and return the raw data.
        This method is called by the base class and should return data directly.
        """
        if tool_name != "get_recent_large_trades":
            return {"error": f"Unsupported tool '{tool_name}'"}

        chain = function_args.get("chain")
        tokenAddress = function_args.get("tokenAddress")

        if not chain:
            return {"error": "Chain parameter is required"}
        if not tokenAddress:
            return {
                "error": "Token address is required. Please provide a valid EVM token contract address starting with '0x'"
            }

        return await self.get_recent_large_trades(
            chain=chain,
            tokenAddress=tokenAddress,
            minUsdAmount=function_args.get("minUsdAmount", 5000),
            filter=function_args.get("filter", "all"),
            limit=function_args.get("limit", 10),
        )
