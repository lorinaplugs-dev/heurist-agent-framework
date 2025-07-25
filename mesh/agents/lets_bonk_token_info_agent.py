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
        self.DEFAULT_LIMIT = 10

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
                "description": "This agent analyzes LetsBonk.fun tokens on Solana using Bitquery API. It tracks tokens about to graduate, provides trading data, price information, identifies top buyers/sellers, OHLCV data, pair addresses, and liquidity information on the Raydium Launchpad.",
                "external_apis": ["Bitquery"],
                "tags": ["LetsBonk", "solana"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/LetsBonk.png",
                "examples": [
                    "Show me top 10 tokens about to graduate on LetsBonk.fun",
                    "Get latest trades for token So11111111111111111111111111111111111111112",
                    "What's the current price of token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "Show me top buyers of token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                    "Show me top sellers of token JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
                    "Get OHLCV data for token MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",
                    "Get pair address for token orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
                    "Get liquidity for pool address 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """You are a specialized assistant that analyzes LetsBonk.fun tokens on Solana using Bitquery API. Your capabilities include:

            1. Graduation Tracking: Monitor tokens that are about to graduate on LetsBonk.fun (approaching the graduation threshold)
            2. Trading Analysis: Track recent trades for specific tokens on Raydium Launchpad
            3. Price Monitoring: Get latest price information for LetsBonk.fun tokens
            4. Buyer Analysis: Identify top buyers and their trading volumes for specific tokens
            5. Seller Analysis: Identify top sellers and their trading volumes for specific tokens
            6. OHLCV Data: Get Open, High, Low, Close, Volume data for specific tokens
            7. Pair Discovery: Get pair/pool addresses for specific tokens
            8. Liquidity Monitoring: Get current liquidity data for token pairs

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
            - All data is sourced from Bitquery API with real-time updates
            - Default limit is 10 entries to ensure fast responses, user can request more if needed"""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "query_about_to_graduate_tokens",
                    "description": "Get top LetsBonk.fun tokens that are about to graduate. These tokens are close to hitting the graduation threshold and may see increased trading activity.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                                "description": "Number of tokens to return (default 10, max 100)",
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
                                "default": 10,
                                "description": "Number of recent trades to return (default 10)",
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
                                "default": 10,
                                "description": "Number of top buyers to return",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_top_sellers",
                    "description": "Get the top sellers for a specific LetsBonk.fun token on Raydium Launchpad. Shows who has sold the most (by USD volume) and can help identify distribution patterns.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get top sellers for",
                            },
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                                "description": "Number of top sellers to return",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_ohlcv_data",
                    "description": "Get OHLCV (Open, High, Low, Close, Volume) data for a specific LetsBonk.fun token on Raydium Launchpad. Returns candlestick data for technical analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get OHLCV data for",
                            },
                            "limit": {
                                "type": "number",
                                "minimum": 1,
                                "maximum": 100,
                                "default": 10,
                                "description": "Number of time intervals to return",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_pair_address",
                    "description": "Get the pair/pool address for a specific LetsBonk.fun token on Raydium Launchpad. Returns the market address and trading pair information.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "token_address": {
                                "type": "string",
                                "description": "The token mint address to get pair address for",
                            },
                        },
                        "required": ["token_address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "query_liquidity",
                    "description": "Get current liquidity data for a specific token pool address. Returns the amounts of base and quote tokens in the liquidity pool.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "pool_address": {
                                "type": "string",
                                "description": "The pool/market address to get liquidity for",
                            },
                        },
                        "required": ["pool_address"],
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
    async def query_about_to_graduate_tokens(self, limit: int = None, since_date: Optional[str] = None) -> Dict:
        """
        Get top tokens that are about to graduate on LetsBonk.fun.

        Args:
            limit (int): Number of tokens to return (defaults to 10, max 100)
            since_date (str, optional): ISO date string to filter since this date

        Returns:
            Dict: Dictionary containing tokens about to graduate
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
        elif limit > 100:
            limit = 100

        if not since_date:
            since_date = (datetime.now(timezone.utc) - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")

        query = """query ($limit: Int!, $since_date: DateTime!, $quote_currencies: [String!]!, $graduation_threshold: String!, $dex_program: String!) { Solana { DEXPools(limitBy: { by: Pool_Market_BaseCurrency_MintAddress, count: 1 } limit: { count: $limit } orderBy: { ascending: Pool_Base_PostAmount } where: { Pool: { Base: { PostAmount: { gt: $graduation_threshold } } Dex: { ProgramAddress: { is: $dex_program } } Market: { QuoteCurrency: { MintAddress: { in: $quote_currencies } } } } Transaction: { Result: { Success: true } } Block: { Time: { since: $since_date } } }) { Pool { Market { BaseCurrency { MintAddress Name Symbol } MarketAddress QuoteCurrency { MintAddress Name Symbol } } Dex { ProtocolName ProtocolFamily } Base { PostAmount(maximum: Block_Time) } Quote { PostAmount PriceInUSD PostAmountInUSD } } Block { Time } } } }"""

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
    @with_cache(ttl_seconds=60)
    @with_retry(max_retries=3)
    async def query_latest_trades(self, token_address: str, limit: int = None) -> Dict:
        """
        Get latest trades for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of recent trades to return (defaults to 10)

        Returns:
            Dict: Dictionary containing recent trades
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
        elif limit > 100:
            limit = 100

        query = """query ($token_address: String!, $limit: Int!, $protocol_name: String!) { Solana { DEXTradeByTokens(orderBy: { descending: Block_Time } limit: { count: $limit } where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } } }) { Block { Time } Transaction { Signature } Trade { Market { MarketAddress } Dex { ProtocolName ProtocolFamily } AmountInUSD PriceInUSD Amount Currency { Name Symbol MintAddress } Side { Type Currency { Symbol MintAddress Name } AmountInUSD Amount } } } } }"""

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
    @with_cache(ttl_seconds=30)
    @with_retry(max_retries=3)
    async def query_latest_price(self, token_address: str) -> Dict:
        """
        Get latest price for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address

        Returns:
            Dict: Dictionary containing latest price data
        """
        query = """query ($token_address: String!, $protocol_name: String!) { Solana { DEXTradeByTokens(orderBy: { descending: Block_Time } limit: { count: 1 } where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } } }) { Block { Time } Transaction { Signature } Trade { Market { MarketAddress } Dex { ProtocolName ProtocolFamily } AmountInUSD PriceInUSD Amount Currency { Name Symbol MintAddress } Side { Type Currency { Symbol MintAddress Name } AmountInUSD Amount } } } } }"""

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
    async def query_top_buyers(self, token_address: str, limit: int = None) -> Dict:
        """
        Get top buyers for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of top buyers to return (defaults to 10)

        Returns:
            Dict: Dictionary containing top buyers
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
        elif limit > 100:
            limit = 100

        query = """query ($token_address: String!, $protocol_name: String!, $limit: Int!) { Solana { DEXTradeByTokens(where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } Side: { Type: { is: buy } } } } orderBy: { descendingByField: "buy_volume" } limit: { count: $limit }) { Trade { Currency { MintAddress Name Symbol } } Transaction { Signer } buy_volume: sum(of: Trade_Side_AmountInUSD) } } }"""

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

    @monitor_execution()
    @with_cache(ttl_seconds=300)
    @with_retry(max_retries=3)
    async def query_top_sellers(self, token_address: str, limit: int = None) -> Dict:
        """
        Get top sellers for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of top sellers to return (defaults to 10)

        Returns:
            Dict: Dictionary containing top sellers
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
        elif limit > 100:
            limit = 100

        query = """query ($token_address: String!, $protocol_name: String!, $limit: Int!) { Solana { DEXTradeByTokens(where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } Side: { Type: { is: sell } } } } orderBy: { descendingByField: "sell_volume" } limit: { count: $limit }) { Trade { Currency { MintAddress Name Symbol } } Transaction { Signer } sell_volume: sum(of: Trade_Side_AmountInUSD) } } }"""

        variables = {
            "token_address": token_address,
            "protocol_name": self.RAYDIUM_LAUNCHPAD,
            "limit": limit,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            sellers = result["data"]["Solana"]["DEXTradeByTokens"]
            formatted_sellers = []

            for seller_data in sellers:
                trade = seller_data.get("Trade", {})
                currency = trade.get("Currency", {})
                transaction = seller_data.get("Transaction", {})

                try:
                    sell_volume = float(seller_data.get("sell_volume", 0))
                except (ValueError, TypeError):
                    sell_volume = 0

                formatted_seller = {
                    "seller_address": transaction.get("Signer", ""),
                    "total_sell_volume_usd": sell_volume,
                    "token_info": {
                        "name": currency.get("Name", "Unknown"),
                        "symbol": currency.get("Symbol", "Unknown"),
                        "mint_address": currency.get("MintAddress", ""),
                    },
                }
                formatted_sellers.append(formatted_seller)

            # Calculate some stats
            total_volume = sum(seller["total_sell_volume_usd"] for seller in formatted_sellers)

            return {
                "top_sellers": formatted_sellers,
                "total_count": len(formatted_sellers),
                "total_sell_volume_usd": total_volume,
                "token_address": token_address,
                "stats": {
                    "largest_seller_volume": formatted_sellers[0]["total_sell_volume_usd"] if formatted_sellers else 0,
                    "average_sell_volume": total_volume / len(formatted_sellers) if formatted_sellers else 0,
                },
            }

        return {"top_sellers": [], "total_count": 0, "total_sell_volume_usd": 0}

    @monitor_execution()
    @with_cache(ttl_seconds=180)
    @with_retry(max_retries=3)
    async def query_ohlcv_data(self, token_address: str, limit: int = None) -> Dict:
        """
        Get OHLCV data for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address
            limit (int): Number of time intervals to return (defaults to 10)

        Returns:
            Dict: Dictionary containing OHLCV data
        """
        if limit is None:
            limit = self.DEFAULT_LIMIT
        elif limit > 100:
            limit = 100

        query = """query ($token_address: String!, $protocol_name: String!, $limit: Int!, $wsol_address: String!) { Solana { DEXTradeByTokens(where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } Side: { Currency: { MintAddress: { is: $wsol_address } } } } Transaction: { Result: { Success: true } } } limit: { count: $limit } orderBy: { descendingByField: "Block_Timefield" }) { Block { Timefield: Time(interval: { count: 1, in: minutes }) } Trade { open: Price(minimum: Block_Slot) high: Price(maximum: Trade_Price) low: Price(minimum: Trade_Price) close: Price(maximum: Block_Slot) } volumeInUSD: sum(of: Trade_Side_AmountInUSD) count } } }"""

        variables = {
            "token_address": token_address,
            "protocol_name": self.RAYDIUM_LAUNCHPAD,
            "limit": limit,
            "wsol_address": "So11111111111111111111111111111111111111112",
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            ohlcv_data = result["data"]["Solana"]["DEXTradeByTokens"]
            formatted_ohlcv = []

            for candle_data in ohlcv_data:
                trade = candle_data.get("Trade", {})
                block = candle_data.get("Block", {})

                try:
                    open_price = float(trade.get("open", 0))
                    high_price = float(trade.get("high", 0))
                    low_price = float(trade.get("low", 0))
                    close_price = float(trade.get("close", 0))
                    volume_usd = float(candle_data.get("volumeInUSD", 0))
                    trade_count = int(candle_data.get("count", 0))
                except (ValueError, TypeError):
                    open_price = 0
                    high_price = 0
                    low_price = 0
                    close_price = 0
                    volume_usd = 0
                    trade_count = 0

                formatted_candle = {
                    "timestamp": block.get("Timefield"),
                    "open": open_price,
                    "high": high_price,
                    "low": low_price,
                    "close": close_price,
                    "volume_usd": volume_usd,
                    "trade_count": trade_count,
                }
                formatted_ohlcv.append(formatted_candle)

            # Calculate some basic stats
            total_volume = sum(candle["volume_usd"] for candle in formatted_ohlcv)
            total_trades = sum(candle["trade_count"] for candle in formatted_ohlcv)

            return {
                "ohlcv_data": formatted_ohlcv,
                "total_count": len(formatted_ohlcv),
                "token_address": token_address,
                "stats": {
                    "total_volume_usd": total_volume,
                    "total_trades": total_trades,
                    "average_volume_per_interval": total_volume / len(formatted_ohlcv) if formatted_ohlcv else 0,
                    "average_trades_per_interval": total_trades / len(formatted_ohlcv) if formatted_ohlcv else 0,
                },
            }

        return {"ohlcv_data": [], "total_count": 0}

    @monitor_execution()
    @with_cache(ttl_seconds=600)
    @with_retry(max_retries=3)
    async def query_pair_address(self, token_address: str) -> Dict:
        """
        Get pair address for a specific LetsBonk.fun token.

        Args:
            token_address (str): Token mint address

        Returns:
            Dict: Dictionary containing pair address information
        """
        query = """query ($token_address: String!, $protocol_name: String!) { Solana { DEXTradeByTokens(where: { Trade: { Dex: { ProtocolName: { is: $protocol_name } } Currency: { MintAddress: { is: $token_address } } } } limit: { count: 10 }) { Trade { Market { MarketAddress } Currency { Name Symbol MintAddress } Side { Currency { Name Symbol MintAddress } } } count } } }"""

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
                return {"error": f"No pair found for token {token_address}"}

            # Group by market address and currency pairs
            pairs = {}
            for trade_data in trades:
                trade = trade_data.get("Trade", {})
                market = trade.get("Market", {})
                currency = trade.get("Currency", {})
                side_currency = trade.get("Side", {}).get("Currency", {})

                market_address = market.get("MarketAddress", "")
                if market_address:
                    if market_address not in pairs:
                        pairs[market_address] = {
                            "market_address": market_address,
                            "base_currency": {
                                "name": currency.get("Name", "Unknown"),
                                "symbol": currency.get("Symbol", "Unknown"),
                                "mint_address": currency.get("MintAddress", ""),
                            },
                            "quote_currency": {
                                "name": side_currency.get("Name", "Unknown"),
                                "symbol": side_currency.get("Symbol", "Unknown"),
                                "mint_address": side_currency.get("MintAddress", ""),
                            },
                            "trade_count": 0,
                        }

                    # Fix: Ensure count is converted to int before adding
                    try:
                        count_value = int(trade_data.get("count", 0))
                    except (ValueError, TypeError):
                        count_value = 0

                    pairs[market_address]["trade_count"] += count_value

            formatted_pairs = list(pairs.values())
            # Sort by trade count descending
            formatted_pairs.sort(key=lambda x: x["trade_count"], reverse=True)

            return {
                "pairs": formatted_pairs,
                "total_pairs": len(formatted_pairs),
                "token_address": token_address,
                "primary_pair": formatted_pairs[0] if formatted_pairs else None,
            }

        return {"pairs": [], "total_pairs": 0}

    @monitor_execution()
    @with_cache(ttl_seconds=60)  # Short cache for liquidity as it changes frequently
    @with_retry(max_retries=3)
    async def query_liquidity(self, pool_address: str) -> Dict:
        """
        Get liquidity data for a specific pool address.

        Args:
            pool_address (str): Pool/market address

        Returns:
            Dict: Dictionary containing liquidity data
        """
        query = """query ($pool_address: String!) { Solana { DEXPools(where: { Pool: { Market: { MarketAddress: { is: $pool_address } } } Transaction: { Result: { Success: true } } } orderBy: { descending: Block_Time } limit: { count: 1 }) { Pool { Base { PostAmount } Quote { PostAmount } Market { BaseCurrency { MintAddress Name Symbol } QuoteCurrency { MintAddress Name Symbol } } } Block { Time } } } }"""

        variables = {
            "pool_address": pool_address,
        }

        result = await self._execute_query(query, variables)

        if "error" in result:
            return result

        if "data" in result and "Solana" in result["data"]:
            pools = result["data"]["Solana"]["DEXPools"]

            if not pools:
                return {"error": f"No liquidity data found for pool {pool_address}"}

            pool_data = pools[0]
            pool = pool_data.get("Pool", {})
            market = pool.get("Market", {})
            base_currency = market.get("BaseCurrency", {})
            quote_currency = market.get("QuoteCurrency", {})
            base = pool.get("Base", {})
            quote = pool.get("Quote", {})

            try:
                base_amount = float(base.get("PostAmount", 0))
                quote_amount = float(quote.get("PostAmount", 0))
            except (ValueError, TypeError):
                base_amount = 0
                quote_amount = 0

            liquidity_data = {
                "pool_address": pool_address,
                "last_updated": pool_data.get("Block", {}).get("Time"),
                "base_liquidity": {
                    "amount": base_amount,
                    "currency": {
                        "name": base_currency.get("Name", "Unknown"),
                        "symbol": base_currency.get("Symbol", "Unknown"),
                        "mint_address": base_currency.get("MintAddress", ""),
                    },
                },
                "quote_liquidity": {
                    "amount": quote_amount,
                    "currency": {
                        "name": quote_currency.get("Name", "Unknown"),
                        "symbol": quote_currency.get("Symbol", "Unknown"),
                        "mint_address": quote_currency.get("MintAddress", ""),
                    },
                },
                "total_liquidity_summary": {
                    "base_symbol": base_currency.get("Symbol", "Unknown"),
                    "quote_symbol": quote_currency.get("Symbol", "Unknown"),
                    "base_amount": base_amount,
                    "quote_amount": quote_amount,
                },
            }

            return {"liquidity_data": liquidity_data, "pool_address": pool_address}

        return {"error": f"No liquidity data found for pool {pool_address}"}

    async def _handle_tool_logic(
        self, tool_name: str, function_args: dict, session_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Handle execution of specific tools and return the raw data.
        This method is required by the MeshAgent abstract base class.
        """
        if tool_name == "query_about_to_graduate_tokens":
            limit = function_args.get("limit", self.DEFAULT_LIMIT)
            since_date = function_args.get("since_date")
            result = await self.query_about_to_graduate_tokens(limit=limit, since_date=since_date)
        elif tool_name == "query_latest_trades":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            limit = function_args.get("limit", self.DEFAULT_LIMIT)
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
            limit = function_args.get("limit", self.DEFAULT_LIMIT)
            result = await self.query_top_buyers(token_address=token_address, limit=limit)
        elif tool_name == "query_top_sellers":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            limit = function_args.get("limit", self.DEFAULT_LIMIT)
            result = await self.query_top_sellers(token_address=token_address, limit=limit)
        elif tool_name == "query_ohlcv_data":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            limit = function_args.get("limit", self.DEFAULT_LIMIT)
            result = await self.query_ohlcv_data(token_address=token_address, limit=limit)
        elif tool_name == "query_pair_address":
            token_address = function_args.get("token_address")
            if not token_address:
                return {"error": "Missing 'token_address' parameter"}
            result = await self.query_pair_address(token_address=token_address)
        elif tool_name == "query_liquidity":
            pool_address = function_args.get("pool_address")
            if not pool_address:
                return {"error": "Missing 'pool_address' parameter"}
            result = await self.query_liquidity(pool_address=pool_address)
        else:
            return {"error": f"Unsupported tool '{tool_name}'"}

        if errors := self._handle_error(result):
            return errors

        return result
