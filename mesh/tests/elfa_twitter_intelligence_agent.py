import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.elfa_twitter_intelligence_agent import ElfaTwitterIntelligenceAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = ElfaTwitterIntelligenceAgent()
    try:
        agent_input_mentions_tool = {
            "tool": "search_mentions",
            "tool_arguments": {"keywords": ["bitcoin", "solana", "ethereum"], "days_ago": 30, "limit": 25},
            "query": "Search for crypto mentions using tool arguments",
        }
        agent_output_mentions_tool = await agent.handle_message(agent_input_mentions_tool)

        agent_input_mentions_query = {
            "query": "Search for mentions of bitcoin, solana, and ethereum in the last 30 days"
        }
        agent_output_mentions_query = await agent.handle_message(agent_input_mentions_query)

        agent_input_account_tool = {
            "tool": "search_account",
            "tool_arguments": {"username": "heurist_ai", "days_ago": 30, "limit": 20},
            "query": "Analyze account using tool arguments",
        }
        agent_output_account_tool = await agent.handle_message(agent_input_account_tool)

        agent_input_account_query = {
            "query": "Analyze the Twitter account @heurist_ai and show me their recent activity"
        }
        agent_output_account_query = await agent.handle_message(agent_input_account_query)

        agent_input_trending_tool = {
            "tool": "get_trending_tokens",
            "tool_arguments": {"time_window": "24h"},
            "query": "Get trending tokens using tool arguments",
        }
        agent_output_trending_tool = await agent.handle_message(agent_input_trending_tool)

        agent_input_trending_query = {"query": "What are the trending tokens on Twitter in the last 24 hours?"}
        agent_output_trending_query = await agent.handle_message(agent_input_trending_query)

        api_dance_test_query = {"tweet_id": "1913624766793289972"}
        api_dance_output = await agent.get_tweet_detail(api_dance_test_query["tweet_id"])

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        output_file = script_dir / f"{current_file}_example.yaml"

        yaml_content = {
            "test_1_search_mentions_tool": {"input": agent_input_mentions_tool, "output": agent_output_mentions_tool},
            "test_2_search_mentions_query": {
                "input": agent_input_mentions_query,
                "output": agent_output_mentions_query,
            },
            "test_3_search_account_tool": {"input": agent_input_account_tool, "output": agent_output_account_tool},
            "test_4_search_account_query": {"input": agent_input_account_query, "output": agent_output_account_query},
            "test_5_get_trending_tokens_tool": {
                "input": agent_input_trending_tool,
                "output": agent_output_trending_tool,
            },
            "test_6_get_trending_tokens_query": {
                "input": agent_input_trending_query,
                "output": agent_output_trending_query,
            },
            "test_7_apidance_tweet_detail": {"input": api_dance_test_query, "output": api_dance_output},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False, indent=2)

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
