import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from decorators import monitor_execution, with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)
load_dotenv()


class LetsBonkTokenInfoAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("BITQUERY_API_KEY")
        if not self.api_key:
            raise ValueError("BITQUERY_API_KEY environment variable is required")
        self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        self.bitquery_url = "https://streaming.bitquery.io/eap"

        # LetsBonk.fun specific constants
        self.LETSBONK_DEX_PROGRAM = "LanMV9sAd7wArD4vJFi2qDdfnVhFxYSUg6eADduJ3uj"
        self.RAYDIUM_LAUNCHPAD = "raydium_launchpad"
        self.GRADUATION_THRESHOLD = "206900000"

        # Supported quote currencies for LetsBonk
        self.QUOTE_CURRENCIES = [
            "11111111111111111111111111111111",  # Native SOL
            "So11111111111111111111111111111111111111112",  # Wrapped SOL
        ]

        self.metadata.update(
            {
                "name": "LetsBonk Token Info Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent analyzes LetsBonk.fun tokens on Solana using Bitquery API. It tracks tokens about to graduate, provides trading data, price information, and identifies top buyers on the Raydium Launchpad.",
                "external_apis": ["Bitquery"],
                "tags": ["LetsBonk", "solana"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/LetsBonk.png",
                "examples": [
                    "Show me top 100 tokens about to graduate on LetsBonk.fun",
                    "Get latest trades for token ABC123...",
                    "What's the current price of token XYZ789...",
                    "Show me top buyers of token DEF456...",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """You are a specialized assistant that analyzes LetsBonk.fun tokens on Solana using Bitquery API. Your capabilities include:

1. Graduation Tracking: Monitor tokens that are about to graduate on LetsBonk.fun (approaching the graduation threshold)
2. Trading Analysis: Track recent trades for specific tokens on Raydium Launchpad
3. Price Monitoring: Get latest price information for LetsBonk.fun tokens
4. Buyer Analysis: Identify top buyers and their trading volumes for specific tokens

LetsBonk.fun Context:
- LetsBonk.fun is a token launchpad on Solana similar to Pump.fun
- Tokens "graduate" when they reach a certain market cap threshold
- Trading happens on Raydium Launchpad
- Graduation threshold is approximately 206,900,000 base tokens

Guidelines:
- Present data in a clear, concise, and data-driven manner
- Focus on actionable insights for traders and investors
- For token addresses, use this format: [Mint Address](https://solscan.io/token/Mint_Address)
- Use natural language in responses
- Highlight tokens close to graduation as they may see increased trading activity
- All data is sourced from Bitquery API with real-time updates"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_about_to_graduate_tokens",
                    "description": "Get top 100 LetsBonk.fun tokens that are about to graduate. These tokens are close to hitting the graduation threshold and may see increased trading activity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 100,
                                "description": "Number of tokens to return (max 100)",
                            },
                            "since_date": {
                                "type": "string",
                                "description": "ISO date string to filter tokens since this date (e.g., '2025-07-11T13:45:00Z'). Defaults to 7 days ago.",
                            },
                        },
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_latest_trades",
                    "description": "Get the most recent trades for a specific LetsBonk.fun token on Raydium Launchpad. Useful for tracking trading activity and price movements.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get trades for",
                            },
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 50,
                                "description": "Number of recent trades to return",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_latest_price",
                    "description": "Get the most recent price data for a specific LetsBonk.fun token on Raydium Launchpad. Returns the latest trade price and transaction details.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get price for",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_top_buyers",
                    "description": "Get the top buyers for a specific LetsBonk.fun token on Raydium Launchpad. Shows who has bought the most (by USD volume) and can help identify whales and smart money.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get top buyers for",
                            },
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 100,
                                "description": "Number of top buyers to return",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
        ]

    async def _execute_query(self, query: str, variables: Dict = None) -> Dict:
        """
        Execute a GraphQL query against the Bitquery API using the base class's _api_request method.

        Args:
            query (str): GraphQL query to execute
            variables (Dict, optional): Variables for the query

        Returns:
            Dict: Query results
        """
        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            result = await self._api_request(
                url=self.bitquery_url, method="POST", headers=self.headers, json_data=payload
            )

            if "error" in result:
                return result

            if "errors" in result:
                error_messages = [error.get("message", "Unknown error") for error in result["errors"]]
                return {"error": f"GraphQL errors: {', '.join(error_messages)}"}

            return result

        except Exception as e:
            logger.error(f"Error in _execute_query: {str(e)}")
            return {"error": f"Query execution failed: {str(e)}"}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_about_to_graduate_tokens(self, limit: int = 100, since_date: Optional[str] = None) -> Dict:
        """
        Get top tokens that are about to graduate on LetsBonk.fun.

        Args:
            limit (int): Number of tokens to return (max 100)
            since_date (str, optional): ISO date string to filter since this date

        Returns:
            Dict: Dictionary containing tokens about to graduate
        """
        if limit > 100:
            limit = 100

        # Default to 7 days ago if no since_date provided
        if not since_date:
            since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

        query = """
        query ($limit: Int!, $since_date: DateTime!, $quote_currencies: [String!]!, $graduation_threshold: String!, $dex_program: String!) {
          Solana {
            DEXPools(
              limitBy: { by: Pool_Market_BaseCurrency_MintAddress, count: 1 }
              limit: { count: $limit }
              orderBy: { ascending: Pool_Base_PostAmount }
              where: {
                Pool: {
                  Base: { PostAmount: { gt: $graduation_threshold } }
                  Dex: {
                    ProgramAddress: { is: $dex_program }
                  }
                  Market: {
                    QuoteCurrency: {
                      MintAddress: { in: $quote_currencies }
                    }
                  }
                }
                Transaction: { Result: { Success: true } }
                Block: { Time: { since: $since_date } }
              }
            ) {
              Pool {
                Market {
                  BaseCurrency {
                    MintAddress
                    Name
                    Symbol
                  }
                  MarketAddress
                  QuoteCurrency {
                    MintAddress
                    Name
                    Symbol
                  }
                }
                Dex {
                  ProtocolName
                  ProtocolFamily
                }
                Base {
                  PostAmount(maximum: Block_Time)
                }
                Quote {
                  PostAmount
                  PriceInUSD
                  PostAmountInUSD
                }
              }
              Block {
                Time
              }
            }
          }
        }
        """

        variables = {
            "limit": limit,
            "since_date": since_date,
            "quote_currencies": self.QUOTE_CURRENCIES,
            "graduation_threshold": self.GRADUATION_THRESHOLD,
            "dex_program": self.LETSBONK_DEX_PROGRAM,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            pools = result["data"]["Solana"]["DEXPools"]
            formatted_tokens = []

            for pool_data in pools:
                pool = pool_data.get("Pool", {})
                market = pool.get("Market", {})
                base_currency = market.get("BaseCurrency", {})
                quote_currency = market.get("QuoteCurrency", {})
                dex = pool.get("Dex", {})
                base = pool.get("Base", {})
                quote = pool.get("Quote", {})

                try:
                    base_amount = float(base.get("PostAmount", 0))
                    quote_price_usd = float(quote.get("PriceInUSD", 0))
                    quote_amount_usd = float(quote.get("PostAmountInUSD", 0))
                except (ValueError, TypeError):
                    base_amount = 0
                    quote_price_usd = 0
                    quote_amount_usd = 0

                # Calculate graduation progress (assuming max graduation amount)
                graduation_progress = (base_amount / float(self.GRADUATION_THRESHOLD)) * 100 if base_amount > 0 else 0

                formatted_token = {
                    "token_info": {
                        "name": base_currency.get("Name", "Unknown"),
                        "symbol": base_currency.get("Symbol", "Unknown"),
                        "mint_address": base_currency.get("MintAddress", ""),
                    },
                    "market_info": {
                        "market_address": market.get("MarketAddress", ""),
                        "quote_currency": {
                            "name": quote_currency.get("Name", "Unknown"),
                            "symbol": quote_currency.get("Symbol", "Unknown"),
                            "mint_address": quote_currency.get("MintAddress", ""),
                        },
                    },
                    "pool_data": {
                        "base_amount": base_amount,
                        "quote_price_usd": quote_price_usd,
                        "quote_amount_usd": quote_amount_usd,
                        "graduation_progress_percent": graduation_progress,
                    },
                    "dex_info": {
                        "protocol_name": dex.get("ProtocolName", "Unknown"),
                        "protocol_family": dex.get("ProtocolFamily", "Unknown"),
                    },
                    "last_updated": pool_data.get("Block", {}).get("Time"),
                }
                formatted_tokens.append(formatted_token)

            return {
                "tokens": formatted_tokens,
                "total_count": len(formatted_tokens),
                "graduation_threshold": self.GRADUATION_THRESHOLD,
                "since_date": since_date,
            }

        return {"tokens": [], "total_count": 0}

    @monitor_execution()
    @with_cache(ttl_seconds=60)  # Shorter cache for recent trades
    @with_retry(max_retries=3)
    async def query_latest_trades(self, token_address: str, limit: int = 50) -> Dict:
        """
        Get latest trades for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of recent trades to return

        Returns:
            Dict: Dictionary containing recent trades
        """
        query = """
        query ($token_address: String!, $limit: Int!, $protocol_name: String!) {
          Solana {
            DEXTradeByTokens(
              orderBy: { descending: Block_Time }
              limit: { count: $limit }
              where: {
                Trade: {
                  Dex: { ProtocolName: { is: $protocol_name } }
                  Currency: { MintAddress: { is: $token_address } }
                }
              }
            ) {
              Block {
                Time
              }
              Transaction {
                Signature
              }
              Trade {
                Market {
                  MarketAddress
                }
                Dex {
                  ProtocolName
                  ProtocolFamily
                }
                AmountInUSD
                PriceInUSD
                Amount
                Currency {
                  Name
                  Symbol
                  MintAddress
                }
                Side {
                  Type
                  Currency {
                    Symbol
                    MintAddress
                    Name
                  }
                  AmountInUSD
                  Amount
                }
              }
            }
          }
        }
        """

        variables = {
            "token_address": token_address,
            "limit": limit,
            "protocol_name": self.RAYDIUM_LAUNCHPAD,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            trades = result["data"]["Solana"]["DEXTradeByTokens"]
            formatted_trades = []

            for trade_data in trades:
                trade = trade_data.get("Trade", {})
                currency = trade.get("Currency", {})
                side = trade.get("Side", {})
                side_currency = side.get("Currency", {})
                market = trade.get("Market", {})
                dex = trade.get("Dex", {})

                try:
                    amount_usd = float(trade.get("AmountInUSD", 0))
                    price_usd = float(trade.get("PriceInUSD", 0))
                    amount = float(trade.get("Amount", 0))
                    side_amount_usd = float(side.get("AmountInUSD", 0))
                    side_amount = float(side.get("Amount", 0))
                except (ValueError, TypeError):
                    amount_usd = 0
                    price_usd = 0
                    amount = 0
                    side_amount_usd = 0
                    side_amount = 0

                formatted_trade = {
                    "timestamp": trade_data.get("Block", {}).get("Time"),
                    "transaction_signature": trade_data.get("Transaction", {}).get("Signature"),
                    "market_address": market.get("MarketAddress", ""),
                    "dex_info": {
                        "protocol_name": dex.get("ProtocolName", "Unknown"),
                        "protocol_family": dex.get("ProtocolFamily", "Unknown"),
                    },
                    "trade_data": {
                        "amount_usd": amount_usd,
                        "price_usd": price_usd,
                        "amount": amount,
                        "currency": {
                            "name": currency.get("Name", "Unknown"),
                            "symbol": currency.get("Symbol", "Unknown"),
                            "mint_address": currency.get("MintAddress", ""),
                        },
                    },
                    "side_data": {
                        "type": side.get("Type", "Unknown"),
                        "amount_usd": side_amount_usd,
                        "amount": side_amount,
                        "currency": {
                            "name": side_currency.get("Name", "Unknown"),
                            "symbol": side_currency.get("Symbol", "Unknown"),
                            "mint_address": side_currency.get("MintAddress", ""),
                        },
                    },
                }
                formatted_trades.append(formatted_trade)

            return {
                "trades": formatted_trades,
                "total_count": len(formatted_trades),
                "token_address": token_address,
            }

        return {"trades": [], "total_count": 0}

    @monitor_execution()
    @with_cache(ttl_seconds=30)  # Very short cache for price data
    @with_retry(max_retries=3)
    async def query_latest_price(self, token_address: str) -> Dict:
        """
        Get latest price for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address

        Returns:
            Dict: Dictionary containing latest price data
        """
        query = """
        query ($token_address: String!, $protocol_name: String!) {
          Solana {
            DEXTradeByTokens(
              orderBy: { descending: Block_Time }
              limit: { count: 1 }
              where: {
                Trade: {
                  Dex: { ProtocolName: { is: $protocol_name } }
                  Currency: { MintAddress: { is: $token_address } }
                }
              }
            ) {
              Block {
                Time
              }
              Transaction {
                Signature
              }
              Trade {
                Market {
                  MarketAddress
                }
                Dex {
                  ProtocolName
                  ProtocolFamily
                }
                AmountInUSD
                PriceInUSD
                Amount
                Currency {
                  Name
                  Symbol
                  MintAddress
                }
                Side {
                  Type
                  Currency {
                    Symbol
                    MintAddress
                    Name
                  }
                  AmountInUSD
                  Amount
                }
              }
            }
          }
        }
        """

        variables = {
            "token_address": token_address,
            "protocol_name": self.RAYDIUM_LAUNCHPAD,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            trades = result["data"]["Solana"]["DEXTradeByTokens"]

            if not trades:
                return {"error": f"No price data found for token {token_address}"}

            trade_data = trades[0]
            trade = trade_data.get("Trade", {})
            currency = trade.get("Currency", {})
            side = trade.get("Side", {})
            side_currency = side.get("Currency", {})
            market = trade.get("Market", {})
            dex = trade.get("Dex", {})

            try:
                amount_usd = float(trade.get("AmountInUSD", 0))
                price_usd = float(trade.get("PriceInUSD", 0))
                amount = float(trade.get("Amount", 0))
                side_amount_usd = float(side.get("AmountInUSD", 0))
                side_amount = float(side.get("Amount", 0))
            except (ValueError, TypeError):
                amount_usd = 0
                price_usd = 0
                amount = 0
                side_amount_usd = 0
                side_amount = 0

            price_data = {
                "latest_price_usd": price_usd,
                "last_trade_timestamp": trade_data.get("Block", {}).get("Time"),
                "transaction_signature": trade_data.get("Transaction", {}).get("Signature"),
                "market_address": market.get("MarketAddress", ""),
                "dex_info": {
                    "protocol_name": dex.get("ProtocolName", "Unknown"),
                    "protocol_family": dex.get("ProtocolFamily", "Unknown"),
                },
                "token_info": {
                    "name": currency.get("Name", "Unknown"),
                    "symbol": currency.get("Symbol", "Unknown"),
                    "mint_address": currency.get("MintAddress", ""),
                },
                "trade_details": {
                    "amount_usd": amount_usd,
                    "amount": amount,
                    "side_type": side.get("Type", "Unknown"),
                    "side_amount_usd": side_amount_usd,
                    "side_amount": side_amount,
                    "side_currency": {
                        "name": side_currency.get("Name", "Unknown"),
                        "symbol": side_currency.get("Symbol", "Unknown"),
                        "mint_address": side_currency.get("MintAddress", ""),
                    },
                },
            }

            return {"price_data": price_data, "token_address": token_address}

        return {"error": f"No price data found for token {token_address}"}

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_top_buyers(self, token_address: str, limit: int = 100) -> Dict:
        """
        Get top buyers for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of top buyers to return

        Returns:
            Dict: Dictionary containing top buyers
        """
        query = """
        query ($token_address: String!, $protocol_name: String!, $limit: Int!) {
          Solana {
            DEXTradeByTokens(
              where: {
                Trade: {
                  Dex: { ProtocolName: { is: $protocol_name } }
                  Currency: { MintAddress: { is: $token_address } }
                  Side: { Type: { is: buy } }
                }
              }
              orderBy: { descendingByField: "buy_volume" }
              limit: { count: $limit }
            ) {
              Trade {
                Currency {
                  MintAddress
                  Name
                  Symbol
                }
              }
              Transaction {
                Signer
              }
              buy_volume: sum(of: Trade_Side_AmountInUSD)
            }
          }
        }
        """

        variables = {
            "token_address": token_address,
            "protocol_name": self.RAYDIUM_LAUNCHPAD,
            "limit": limit,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            buyers = result["data"]["Solana"]["DEXTradeByTokens"]
            formatted_buyers = []

            for buyer_data in buyers:
                trade = buyer_data.get("Trade", {})
                currency = trade.get("Currency", {})
                transaction = buyer_data.get("Transaction", {})

                try:
                    buy_volume = float(buyer_data.get("buy_volume", 0))
                except (ValueError, TypeError):
                    buy_volume = 0

                formatted_buyer = {
                    "buyer_address": transaction.get("Signer", ""),
                    "total_buy_volume_usd": buy_volume,
                    "token_info": {
                        "name": currency.get("Name", "Unknown"),
                        "symbol": currency.get("Symbol", "Unknown"),
                        "mint_address": currency.get("MintAddress", ""),
                    },
                }
                formatted_buyers.append(formatted_buyer)

            # Calculate some stats
            total_volume = sum(buyer["total_buy_volume_usd"] for buyer in formatted_buyers)

            return {
                "top_buyers": formatted_buyers,
                "total_count": len(formatted_buyers),
                "total_buy_volume_usd": total_volume,
                "token_address": token_address,
                "stats": {
                    "largest_buyer_volume": formatted_buyers[0]["total_buy_volume_usd"] if formatted_buyers else 0,
                    "average_buy_volume": total_volume / len(formatted_buyers) if formatted_buyers else 0,
                },
            }

        return {"top_buyers": [], "total_count": 0, "total_buy_volume_usd": 0}

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle execution of specific tools and return the raw data.
        This method is required by the MeshAgent abstract base class.
        """
        if tool_name == "query_about_to_graduate_tokens":
            limit = function_args.get("limit", 100)
            since_date = function_args.get("since_date")
            result = await self.query_about_to_graduate_tokens(limit=limit, since_date=since_date)
        elif tool_name == "query_latest_trades":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            limit = function_args.get("limit", 50)
            result = await self.query_latest_trades(token_address=token_address, limit=limit)
        elif tool_name == "query_latest_price":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            result = await self.query_latest_price(token_address=token_address)
        elif tool_name == "query_top_buyers":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            limit = function_args.get("limit", 100)
            result = await self.query_top_buyers(token_address=token_address, limit=limit)
        else:
            return {"error": f"Unsupported tool '{tool_name}'"}

        if errors := self._handle_error(result):
            return errors

        return result
