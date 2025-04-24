import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Load environment variables and include project root in path
load_dotenv()
sys.path.append(str(Path(__file__).parent.parent.parent))

from mesh.agents.cookie_project_info_agent import CookieProjectInfoAgent  # noqa: E402


async def run_agent():
    agent = CookieProjectInfoAgent()
    try:
        test_cases = {
            "trending_projects": {
                "input": {"query": "Show me the top AI projects by mindshare", "raw_data_only": False},
            },
            "project_details": {
                "input": {"query": "Get details for Heurist project", "raw_data_only": False},
            },
            "btc_mindshare_30d": {
                "input": {"query": "Get mindshare data for bitcoin for the past 30 days", "raw_data_only": False},
            },
            "btc_mindshare_15d": {
                "input": {
                    "query": "Show me the mindshare trend for Bitcoin for the last 15 days",
                    "raw_data_only": False,
                },
            },
            "tool_project_details": {
                "input": {"tool": "get_project_details", "tool_arguments": {"slug": "heurist"}, "raw_data_only": True},
            },
            "tool_search_projects": {
                "input": {
                    "tool": "search_projects",
                    "tool_arguments": {"search_query": "DeFi", "limit": 5},
                    "raw_data_only": True,
                },
            },
            "tool_mindshare_graph_7d": {
                "input": {
                    "tool": "get_mindshare_graph",
                    "tool_arguments": {"project_slug": "heurist"},
                    "raw_data_only": True,
                },
            },
            "tool_mindshare_graph_30d": {
                "input": {
                    "tool": "get_mindshare_graph",
                    "tool_arguments": {"project_slug": "heurist", "days": 30},
                    "raw_data_only": True,
                },
            },
            "tool_mindshare_leaderboard": {
                "input": {
                    "tool": "get_mindshare_leaderboard",
                    "tool_arguments": {"timeframe": 2, "sector_slug": "ai", "sort_by": "mindshare"},
                    "raw_data_only": False,
                },
            },
            "growing_projects": {
                "input": {"query": "Find fast-growing projects in the gaming sector", "raw_data_only": False},
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
