import asyncio
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.agents.pond_wallet_analysis_agent import PondWalletAnalysisAgent  # noqa: E402


async def run_agent():
    agent = PondWalletAnalysisAgent()

    try:
        queries = [
            {
                "query": "Analyze Ethereum wallet 0x2B25B37c683F042E9Ae1877bc59A1Bb642Eb1073",
                "platform": "ethereum",
            },
            {
                "query": "What's the trading volume for Solana wallet 8gc59zf1ZQCxzkSuepV8WmuuobHCPpydJ2RLqwXyCASS?",
                "platform": "solana",
            },
            {
                "query": "Check the transaction activity for Base wallet 0x97224Dd2aFB28F6f442E773853F229B3d8A0999a",
                "platform": "base",
            },
        ]

        agent_outputs = {}
        for query in queries:
            agent_input = {
                "query": query["query"],
                "raw_data_only": False,
            }
            print(f"Testing: {query['query']}")
            agent_outputs[query["platform"]] = await agent.handle_message(agent_input)
            time.sleep(2)

        direct_tool_calls = [
            {
                "tool": "analyze_ethereum_wallet",
                "tool_arguments": {"address": "0x73AF3bcf944a6559933396c1577B257e2054D935"},
                "platform": "ethereum",
            },
            {
                "tool": "analyze_solana_wallet",
                "tool_arguments": {"address": "7g275uQ9JuvTa7EC3TERyAsQUGwid9eDYK2JgpSLrmjK"},
                "platform": "solana",
            },
            {
                "tool": "analyze_base_wallet",
                "tool_arguments": {"address": "0x1C0002972259E13dBC5eAF01D108624430c744f9"},
                "platform": "base",
            },
        ]

        for tool_call in direct_tool_calls:
            agent_input = {
                "tool": tool_call["tool"],
                "tool_arguments": tool_call["tool_arguments"],
                "raw_data_only": True,
            }
            print(f"Direct tool call: {tool_call['platform']} wallet {tool_call['tool_arguments']['address']}")
            agent_outputs[f"{tool_call['platform']}_direct"] = await agent.handle_message(agent_input)
            time.sleep(2)

        output_data = {}
        for query in queries:
            output_data[f"{query['platform']}_wallet_natural_language"] = {
                "input": {"query": query["query"], "raw_data_only": False},
                "output": agent_outputs[query["platform"]],
            }
        for tool_call in direct_tool_calls:
            output_data[f"{tool_call['platform']}_wallet_direct"] = {
                "input": {
                    "tool": tool_call["tool"],
                    "tool_arguments": tool_call["tool_arguments"],
                    "raw_data_only": True,
                },
                "output": agent_outputs[f"{tool_call['platform']}_direct"],
            }

        output_file = Path(__file__).parent / f"{Path(__file__).stem}_results.yaml"
        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(output_data, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
