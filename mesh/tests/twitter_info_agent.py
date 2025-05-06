import asyncio
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.twitter_info_agent import TwitterInfoAgent  # noqa: E402

load_dotenv()

# Example tweet IDs for testing
TEST_TWEET_IDS = [
    "1913624766793289972",  # Example tweet ID
    "1914394032169762877",  # Another example tweet ID
]


async def test_tweet_detail_fetching(agent):
    """Test the get_twitter_detail functionality"""
    test_cases = [
        # Direct tool call with just tweet ID
        {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[0]}},
        # Test with another tweet ID
        {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[1]}},
    ]

    results = {}
    for i, test_case in enumerate(test_cases):
        try:
            result = await agent.handle_message(test_case)
            results[f"case_{i + 1}"] = {"input": test_case, "output": result}
            logger.info(f"Tweet detail test case {i + 1} completed successfully")

            await asyncio.sleep(4)
        except Exception as e:
            logger.error(f"Error in tweet detail test case {i + 1}: {e}")
            results[f"case_{i + 1}"] = {"input": test_case, "error": str(e)}

            await asyncio.sleep(4)

    return results


async def test_general_search(agent):
    """Test the get_general_search functionality"""
    test_cases = [
        # Basic search query
        {"tool": "get_general_search", "tool_arguments": {"q": "heurist ai"}},
        # Search with specific hashtag
        {"tool": "get_general_search", "tool_arguments": {"q": "eth"}},
        # Search with cursor for pagination
        {
            "tool": "get_general_search",
            "tool_arguments": {
                "q": "Anthropic MCP",
                "cursor": "",  # Empty cursor for first page
            },
        },
        # Natural language queries
        {"query": "Search for tweets about crypto marketplace"},
        {"query": "Find recent discussions about Heurist AI"},
        {"query": "What are people saying about Vitalik Buterin?"},
    ]

    results = {}
    for i, test_case in enumerate(test_cases):
        try:
            result = await agent.handle_message(test_case)
            results[f"case_{i + 1}"] = {"input": test_case, "output": result}
            logger.info(f"General search test case {i + 1} completed successfully")

            await asyncio.sleep(4)
        except Exception as e:
            logger.error(f"Error in general search test case {i + 1}: {e}")
            results[f"case_{i + 1}"] = {"input": test_case, "error": str(e)}

            await asyncio.sleep(4)

    return results


async def run_agent():
    agent = TwitterInfoAgent()
    try:
        # Test tweet detail functionality
        logger.info("Testing tweet detail functionality...")
        tweet_detail_results = await test_tweet_detail_fetching(agent)

        # Test general search functionality
        logger.info("Testing general search functionality...")
        general_search_results = await test_general_search(agent)

        test_cases = [
            # Test cases that would be handled by LLM's understanding
            {"query": "Summarise recent updates of @heurist_ai", "limit": 5},
            {"query": "What has @elonmusk been tweeting lately?", "limit": 5},
            {"query": "Get the recent tweets from cz_binance", "limit": 5},
            {"query": "Give me details of heurist_ai"},
            {"query": "Show the latest updates from elonmusk"},
            {"query": "What vitalikbuterin has been posting about?"},
            {"query": "can you get me info about @jack"},
            {"query": "twitter.com/naval"},
            {"query": "x.com/pmarca"},
            {"query": "I want to see realdonaldtrump's tweets"},
            {"query": "Check out the profile for naval's account"},
            # Direct API calls
            {"tool": "get_user_tweets", "tool_arguments": {"username": "realdonaldtrump", "limit": 5}},
            # Test with a numeric ID (this is a placeholder ID)
            {"query": "Get tweets from user_id:778764142412984320", "limit": 5},
            # Add test cases for tweet detail
            {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[0]}},
            {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[1], "cursor": ""}},
            # Query-based test cases for tweet detail
            {"query": f"Show me the details and replies for tweet {TEST_TWEET_IDS[0]}"},
            {"query": f"Get all information about this tweet: {TEST_TWEET_IDS[1]}"},
            # Query-based search cases
            {"tool": "get_general_search", "tool_arguments": {"q": "ethereum"}},
            {"query": "Search Twitter for discussions about Heurist AI"},
            {"query": "What are people saying about Heurist AI on Twitter?"},
        ]

        api_results = {}
        for i, test_case in enumerate(test_cases):
            logger.info(f"Running test case {i + 1}: {test_case}")
            try:
                result = await agent.handle_message(test_case)
                api_results[f"case_{i + 1}"] = {"input": test_case, "output": result}

                await asyncio.sleep(4)
            except Exception as e:
                logger.error(f"Error in test case {i + 1}: {e}")
                api_results[f"case_{i + 1}"] = {"input": test_case, "error": str(e)}
                #
                await asyncio.sleep(4)

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "tweet_detail_tests": tweet_detail_results,
            "general_search_tests": general_search_results,
            "api_call_tests": api_results,
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        logger.info(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
