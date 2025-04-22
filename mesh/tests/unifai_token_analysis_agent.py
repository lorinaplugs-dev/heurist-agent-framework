import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.unifai_token_analysis_agent import UnifaiTokenAnalysisAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = UnifaiTokenAnalysisAgent()
    try:
        # Test GMGN trending tokens with natural language query
        agent_input_trend = {
            "query": "Show me trending tokens on GMGN for the last 24 hours",
            "raw_data_only": False,
        }
        print(f"Testing natural language query: {agent_input_trend['query']}")
        agent_output_trend = await agent.handle_message(agent_input_trend)

        # Test direct tool call for GMGN trends
        agent_input_trend_direct = {
            "tool": "get_gmgn_trend",
            "tool_arguments": {"time_window": "4h", "limit": 10},
            "raw_data_only": True,
        }
        print("Testing direct tool call for GMGN trends")
        agent_output_trend_direct = await agent.handle_message(agent_input_trend_direct)

        # Test GMGN token info with natural language query
        agent_input_token_info = {
            "query": "Get token information for 0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599 on Ethereum",
            "raw_data_only": False,
        }
        print(f"Testing natural language query: {agent_input_token_info['query']}")
        agent_output_token_info = await agent.handle_message(agent_input_token_info)

        # Test direct tool call for GMGN token info
        agent_input_token_info_direct = {
            "tool": "get_gmgn_token_info",
            "tool_arguments": {
                "chain": "eth",
                "address": "0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2",  # WETH
            },
            "raw_data_only": True,
        }
        print("Testing direct tool call for GMGN token info")
        agent_output_token_info_direct = await agent.handle_message(agent_input_token_info_direct)

        # Test token analysis with natural language query
        agent_input_analysis = {
            "query": "Analyze ETH token for me",
            "raw_data_only": False,
        }
        print(f"Testing natural language query: {agent_input_analysis['query']}")
        agent_output_analysis = await agent.handle_message(agent_input_analysis)

        # Test direct tool call for token analysis
        agent_input_analysis_direct = {
            "tool": "analyze_token",
            "tool_arguments": {"ticker": "BTC"},
            "raw_data_only": True,
        }
        print("Testing direct tool call for token analysis")
        agent_output_analysis_direct = await agent.handle_message(agent_input_analysis_direct)

        # Save the test inputs and outputs to a YAML file for further inspection
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_results"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "gmgn_trend_natural_language": {"input": agent_input_trend, "output": agent_output_trend},
            "gmgn_trend_direct": {"input": agent_input_trend_direct, "output": agent_output_trend_direct},
            "gmgn_token_info_natural_language": {"input": agent_input_token_info, "output": agent_output_token_info},
            "gmgn_token_info_direct": {
                "input": agent_input_token_info_direct,
                "output": agent_output_token_info_direct,
            },
            "token_analysis_natural_language": {"input": agent_input_analysis, "output": agent_output_analysis},
            "token_analysis_direct": {"input": agent_input_analysis_direct, "output": agent_output_analysis_direct},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
