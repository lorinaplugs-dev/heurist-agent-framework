import logging
import os
from typing import Any, Dict, List

import requests

from decorators import with_cache, with_retry
from mesh.mesh_agent import MeshAgent

logger = logging.getLogger(__name__)


class PondWalletAnalysisAgent(MeshAgent):
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv("POND_API_KEY")
        if not self.api_key:
            raise ValueError("POND_API_KEY environment variable is required")

        self.base_url = "https://broker-service.private.cryptopond.xyz"
        self.headers = {"Content-Type": "application/json"}
        self.model_ids = {
            "ethereum": 20,
            "solana": 24,
            "base": 16,
        }

        self.metadata.update(
            {
                "name": "Pond Wallet Analysis Agent",
                "version": "1.0.0",
                "author": "Heurist team",
                "author_address": "0x7d9d1821d15B9e0b8Ab98A058361233E255E405D",
                "description": "This agent analyzes cryptocurrency wallet activities across Ethereum, Solana, and Base networks using the Cryptopond API.",
                "inputs": [
                    {
                        "name": "query",
                        "description": "Natural language query about wallet analysis",
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
                        "description": "Natural language explanation of wallet analysis",
                        "type": "str",
                    },
                    {"name": "data", "description": "Structured wallet analysis data", "type": "dict"},
                ],
                "external_apis": ["Cryptopond"],
                "tags": ["Wallet Analysis"],
                "image_url": "https://raw.githubusercontent.com/heurist-network/heurist-agent-framework/refs/heads/main/mesh/images/CryptoPond.png",
                "examples": [
                    "Analyze Ethereum wallet 0x2B25B37c683F042E9Ae1877bc59A1Bb642Eb1073",
                    "What's the trading volume for Solana wallet 8gc59zf1ZQCxzkSuepV8WmuuobHCPpydJ2RLqwXyCASS?",
                    "Check the transaction activity for Base wallet 0x97224Dd2aFB28F6f442E773853F229B3d8A0999a",
                ],
            }
        )

    def get_system_prompt(self) -> str:
        return """Analyze crypto wallet activity on Ethereum, Solana, and Base networks. Focus on trading volume, transaction count, gas fees, token diversity, profit and loss, and behavioral patterns over time.

            Identify trends such as accumulation, selling pressure, inactivity, or sudden activity spikes. Pay attention to timing, frequency, and consistency in wallet behavior.

            Present data clearly by formatting large numbers (e.g., 48.5M), emphasizing key insights, and flagging any unusual or suspicious activity.

            Ensure the wallet address format matches the target network (e.g., 0x for Ethereum/Base). Always report missing or incomplete data, and base conclusions strictly on what's available without assumptions."""

    def get_tool_schemas(self) -> List[Dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": "analyze_ethereum_wallet",
                    "description": "Analyze an Ethereum wallet address for trading activity, volume, and transaction metrics over the last 30 days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "Ethereum wallet address (starts with 0x)",
                            },
                        },
                        "required": ["address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_solana_wallet",
                    "description": "Analyze a Solana wallet address for trading activity, volume, and transaction metrics over the last 30 days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "Solana wallet address",
                            },
                        },
                        "required": ["address"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "analyze_base_wallet",
                    "description": "Analyze a Base network wallet address for trading activity, volume, and transaction metrics over the last 30 days",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "address": {
                                "type": "string",
                                "description": "Base wallet address (starts with 0x)",
                            },
                        },
                        "required": ["address"],
                    },
                },
            },
        ]

    # ------------------------------------------------------------------------
    #                      CRYPTOPOND API-SPECIFIC METHODS
    # ------------------------------------------------------------------------
    @with_cache(ttl_seconds=3600)  # Cache for 1 hour
    @with_retry(max_retries=3)
    async def analyze_wallet(self, address: str, network: str) -> Dict:
        """
        Generic method to analyze wallet across different networks
        """
        if network not in self.model_ids:
            return {"error": f"Unsupported network: {network}"}
        model_id = self.model_ids[network]

        try:
            payload = {
                "req_type": "1",
                "access_token": self.api_key,
                "input_keys": [address],
                "model_id": model_id,
            }

            response = requests.post(f"{self.base_url}/predict", headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()

            if result.get("code") != 200 or "resp_items" not in result:
                return {"error": f"API returned unexpected response: {result}"}
            resp_items = result.get("resp_items", [])
            if not resp_items or "analysis_result" not in resp_items[0]:
                return {"error": "No analysis results found in response"}

            analysis_result = resp_items[0]["analysis_result"]
            formatted_result = {
                "network": network,
                "address": address,
                "analysis": analysis_result,
                "updated_at": resp_items[0].get("debug_info", {}).get("UPDATED_AT"),
            }

            return formatted_result

        except requests.exceptions.RequestException as e:
            logger.error(f"Error analyzing {network} wallet: {str(e)}")
            return {"error": f"Failed to analyze wallet: {str(e)}"}
        except Exception as e:
            logger.error(f"Unexpected error during {network} wallet analysis: {str(e)}")
            return {"error": f"Unexpected error: {str(e)}"}

    # ------------------------------------------------------------------------
    #                      TOOL HANDLING LOGIC
    # ------------------------------------------------------------------------
    async def _handle_tool_logic(self, tool_name: str, function_args: dict) -> Dict[str, Any]:
        """
        Handle execution of specific tools and return the raw data
        """
        address = function_args.get("address")
        if not address:
            return {"error": "Missing 'address' parameter"}

        if tool_name == "analyze_ethereum_wallet":
            if not address.startswith("0x"):
                return {"error": "Invalid Ethereum address format. Address should start with '0x'"}
            address = address.lower()
            logger.info(f"Analyzing Ethereum wallet: {address}")
            result = await self.analyze_wallet(address, "ethereum")

        elif tool_name == "analyze_solana_wallet":
            logger.info(f"Analyzing Solana wallet: {address}")
            result = await self.analyze_wallet(address, "solana")

        elif tool_name == "analyze_base_wallet":
            if not address.startswith("0x"):
                return {"error": "Invalid Base address format. Address should start with '0x'"}
            address = address.lower()
            logger.info(f"Analyzing Base wallet: {address}")
            result = await self.analyze_wallet(address, "base")

        else:
            return {"error": f"Unsupported tool: {tool_name}"}

        errors = self._handle_error(result)
        if errors:
            return errors

        return result
