import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.unifai_meteora_info_agent import UnifaiMeteoraInfoAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = UnifaiMeteoraInfoAgent()
    try:
        # Test trending DLMM pools with natural language query
        agent_input_trending = {
            "query": "Show me trending DLMM pools on Meteora",
            "raw_data_only": False,
        }
        # print(f"Testing natural language query: {agent_input_trending['query']}")
        agent_output_trending = await agent.handle_message(agent_input_trending)

        # Test direct tool call for trending DLMM pools
        agent_input_trending_direct = {
            "tool": "get_trending_dlmm_pools",
            "tool_arguments": {"limit": 5},
            "raw_data_only": True,
        }
        # print("Testing direct tool call for trending DLMM pools")
        agent_output_trending_direct = await agent.handle_message(agent_input_trending_direct)

        # Test trending DLMM pools with token pairs filter
        agent_input_trending_filtered = {
            "tool": "get_trending_dlmm_pools",
            "tool_arguments": {"limit": 3, "include_pool_token_pairs": ["SOL/USDC"]},
            "raw_data_only": False,
        }
        # print("Testing trending DLMM pools with SOL/USDC filter")
        agent_output_trending_filtered = await agent.handle_message(agent_input_trending_filtered)

        # Test dynamic AMM pools search with natural language
        agent_input_dynamic = {
            "query": "Find dynamic AMM pools with SOL token",
            "raw_data_only": False,
        }
        # print(f"Testing natural language query: {agent_input_dynamic['query']}")
        agent_output_dynamic = await agent.handle_message(agent_input_dynamic)

        # Test direct tool call for dynamic AMM pools
        agent_input_dynamic_direct = {
            "tool": "search_dynamic_amm_pools",
            "tool_arguments": {
                "limit": 5,
                "include_token_mints": ["So11111111111111111111111111111111111111112"],  # SOL mint
            },
            "raw_data_only": True,
        }
        # print("Testing direct tool call for dynamic AMM pools with SOL")
        agent_output_dynamic_direct = await agent.handle_message(agent_input_dynamic_direct)

        # Test DLMM pools search with natural language
        agent_input_dlmm = {
            "query": "Search for DLMM pools on Meteora",
            "raw_data_only": False,
        }
        # print(f"Testing natural language query: {agent_input_dlmm['query']}")
        agent_output_dlmm = await agent.handle_message(agent_input_dlmm)

        # Test direct tool call for DLMM pools with token pairs
        agent_input_dlmm_direct = {
            "tool": "search_dlmm_pools",
            "tool_arguments": {
                "limit": 5,
                "include_pool_token_pairs": [
                    "So11111111111111111111111111111111111111112-EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
                ],
            },
            "raw_data_only": True,
        }
        # print("Testing direct tool call for DLMM pools with SOL-USDC pair")
        agent_output_dlmm_direct = await agent.handle_message(agent_input_dlmm_direct)

        # Test DLMM pools with both token pairs and mints
        agent_input_dlmm_combined = {
            "tool": "search_dlmm_pools",
            "tool_arguments": {
                "limit": 3,
                "include_token_mints": ["So11111111111111111111111111111111111111112"],
                "include_pool_token_pairs": [],
            },
            "raw_data_only": False,
        }
        # print("Testing DLMM pools with SOL mint filter")
        agent_output_dlmm_combined = await agent.handle_message(agent_input_dlmm_combined)

        # Additional natural language test cases
        test_cases = [
            {"query": "What are the top 5 trending pools on Meteora?"},
            {"query": "Show me pools with high TVL"},
            {"query": "Find liquidity pools for SOL"},
            {"query": "Get me the best DLMM pools"},
        ]

        additional_results = {}
        for i, test_case in enumerate(test_cases):
            # print(f"Testing additional case {i + 1}: {test_case['query']}")
            result = await agent.handle_message(test_case)
            additional_results[f"case_{i + 1}"] = {"input": test_case, "output": result}

        # Save the test inputs and outputs to a YAML file for further inspection
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "trending_dlmm_natural_language": {"input": agent_input_trending, "output": agent_output_trending},
            "trending_dlmm_direct": {"input": agent_input_trending_direct, "output": agent_output_trending_direct},
            "trending_dlmm_filtered": {
                "input": agent_input_trending_filtered,
                "output": agent_output_trending_filtered,
            },
            "dynamic_amm_natural_language": {"input": agent_input_dynamic, "output": agent_output_dynamic},
            "dynamic_amm_direct": {"input": agent_input_dynamic_direct, "output": agent_output_dynamic_direct},
            "dlmm_pools_natural_language": {"input": agent_input_dlmm, "output": agent_output_dlmm},
            "dlmm_pools_direct": {"input": agent_input_dlmm_direct, "output": agent_output_dlmm_direct},
            "dlmm_pools_combined": {"input": agent_input_dlmm_combined, "output": agent_output_dlmm_combined},
            "additional_test_cases": additional_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
