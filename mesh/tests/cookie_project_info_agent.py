import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.agents.cookie_project_info_agent import CookieProjectInfoAgent  # noqa: E402


async def run_agent():
    agent = CookieProjectInfoAgent()
    try:
        test_cases = {
            "twitter_username_query": {
                "input": {
                    "query": "Tell me about the project with Twitter handle @cookiedotfun",
                    "raw_data_only": False,
                },
            },
            "twitter_username_query_2": {
                "input": {
                    "query": "Tell me about the project with Twitter handle @heurist_ai for past 30 days",
                    "raw_data_only": False,
                },
            },
            "contract_address_query": {
                "input": {
                    "query": "Get details for the contract 0xc0041ef357b183448b235a8ea73ce4e4ec8c265f",
                    "raw_data_only": False,
                },
            },
            "tool_twitter_username": {
                "input": {
                    "tool": "get_project_by_twitter_username",
                    "tool_arguments": {"twitter_username": "heurist_ai"},
                    "raw_data_only": True,
                },
            },
            "tool_contract_address": {
                "input": {
                    "tool": "get_project_by_contract_address",
                    "tool_arguments": {"contract_address": "0xc0041ef357b183448b235a8ea73ce4e4ec8c265f"},
                    "raw_data_only": True,
                },
            },
        }

        for case_name, case in test_cases.items():
            input_data = case["input"]
            print(f"Running test: {case_name.replace('_', ' ').title()}")
            output_data = await agent.handle_message(input_data)
            test_cases[case_name]["output"] = output_data

        result_path = Path(__file__).with_name(f"{Path(__file__).stem}_results.yaml")
        with open(result_path, "w", encoding="utf-8") as f:
            yaml.dump(test_cases, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {result_path}")
    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
