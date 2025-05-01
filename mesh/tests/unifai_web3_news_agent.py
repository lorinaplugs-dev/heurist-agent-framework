import asyncio
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.unifai_web3_news_agent import UnifaiWeb3NewsAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = UnifaiWeb3NewsAgent()
    try:
        # Test with a natural language query
        agent_input_query = {
            "query": "What are the latest developments in Web3?",
            "raw_data_only": False,
        }
        print(f"Testing natural language query: {agent_input_query['query']}")
        agent_output_query = await agent.handle_message(agent_input_query)
        print("Result of handle_message (natural language query):")
        print(json.dumps(agent_output_query, indent=2))
        print("\n" + "-" * 80 + "\n")

        # Test with a direct tool call for default parameters
        agent_input_default = {
            "tool": "get_web3_news",
            "tool_arguments": {},
            "raw_data_only": False,
        }
        print("Testing direct tool call with default parameters")
        agent_output_default = await agent.handle_message(agent_input_default)
        print("Result of handle_message (default parameters):")
        print(json.dumps(agent_output_default, indent=2))
        print("\n" + "-" * 80 + "\n")

        # Test with a direct tool call with specific limit
        agent_input_limit = {
            "tool": "get_web3_news",
            "tool_arguments": {"limit": 1},
            "raw_data_only": True,
        }
        print("Testing direct tool call with limit=1 and raw_data_only=True")
        agent_output_limit = await agent.handle_message(agent_input_limit)
        print("Result of handle_message (limit=1, raw_data_only=True):")
        print(json.dumps(agent_output_limit, indent=2))
        print("\n" + "-" * 80 + "\n")

        # Test with a keyword parameter
        agent_input_keyword = {
            "tool": "get_web3_news",
            "tool_arguments": {"keyword": "bitcoin", "limit": 2},
            "raw_data_only": False,
        }
        print("Testing direct tool call with keyword='bitcoin' and limit=2")
        agent_output_keyword = await agent.handle_message(agent_input_keyword)
        print("Result of handle_message (keyword='bitcoin', limit=2):")
        print(json.dumps(agent_output_keyword, indent=2))
        print("\n" + "-" * 80 + "\n")

        # Save the test inputs and outputs to a YAML file for further inspection
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_results"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "natural_language_query": {"input": agent_input_query, "output": agent_output_query},
            "default_parameters": {"input": agent_input_default, "output": agent_output_default},
            "limit_parameter": {"input": agent_input_limit, "output": agent_output_limit},
            "keyword_parameter": {"input": agent_input_keyword, "output": agent_output_keyword},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
