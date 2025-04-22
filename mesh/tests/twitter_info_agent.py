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

async def test_identifier_extraction(agent):
    """Test username and user ID extraction with various query formats"""
    test_queries = [
        # Usernames
        "heurist_ai",
        "Give me details of heurist_ai",
        "Show the latest updates from elonmusk",
        "What vitalikbuterin has been posting about?",
        "can you get me info about @jack",
        "twitter.com/naval",
        "x.com/pmarca",
        "I want to see realdonaldtrump's tweets",
        "Check out the profile for naval's account",
        "Look at the latest from username karpathy",
        "Tweets by chamath",
        "Find the recent activity of handle balajis",
        # User IDs
        "user_id:12345678",
        "userid:12345678",
        "Check twitter ID 12345678",
        "Show me tweets from user:12345678",
        "What's happening with twitter_id:12345678",
        "Get twitter_id: 12345678",
        "12345678",  # Direct ID
        # Ambiguous/difficult queries
        "Get me information about blockchain",  # This should ideally not extract "blockchain" as a username
        "What's happening on Twitter?",  # This should fail to extract any username
        "Show me the latest tech news on Twitter",  # Should not extract any username
    ]

    results = {}
    for query in test_queries:
        # Only test the extraction, don't make API calls
        identifier = agent._extract_username_from_query(query)
        results[query] = identifier
        logger.info(f"Query: '{query}' → Identifier: '{identifier}'")

    return results


async def test_id_pattern_recognition(agent):
    """Test specifically the ID pattern recognition"""
    test_patterns = [
        "user_id:123456789",
        "userid:123456789",
        "id:123456789",
        "user:123456789",
        "twitter id 123456789",
        "twitter_id 123456789",
        "twitter id: 123456789",
        "twitter_id: 123456789",
        "123456789",  # Just the numeric ID
        "Show me tweets from user_id:123456789",
        "Get info about twitter id: 123456789",
    ]

    results = {}
    for pattern in test_patterns:
        id_result = agent._extract_numeric_id(pattern)
        results[pattern] = id_result
        logger.info(f"Pattern: '{pattern}' → ID: '{id_result}'")

    return results


async def test_tweet_detail_fetching(agent):
    """Test the get_twitter_detail functionality"""
    test_cases = [
        # Direct tool call with just tweet ID
        {
            "tool": "get_twitter_detail",
            "tool_arguments": {
                "tweet_id": TEST_TWEET_IDS[0]
            }
        },
        # Test with another tweet ID
        {
            "tool": "get_twitter_detail",
            "tool_arguments": {
                "tweet_id": TEST_TWEET_IDS[1]
            }
        }
    ]

    results = {}
    for i, test_case in enumerate(test_cases):
        try:
            result = await agent.handle_message(test_case)
            results[f"case_{i + 1}"] = {
                "input": test_case,
                "output": result
            }
            logger.info(f"Tweet detail test case {i + 1} completed successfully")
        except Exception as e:
            logger.error(f"Error in tweet detail test case {i + 1}: {e}")
            results[f"case_{i + 1}"] = {
                "input": test_case,
                "error": str(e)
            }

    return results


async def test_general_search(agent):
    """Test the get_general_search functionality"""
    test_cases = [
        # Basic search query
        {
            "tool": "get_general_search",
            "tool_arguments": {
                "q": "heurist ai"
            }
        },
        # Search with specific hashtag
        {
            "tool": "get_general_search",
            "tool_arguments": {
                "q": "eth"
            }
        },
        # Search with cursor for pagination
        {
            "tool": "get_general_search",
            "tool_arguments": {
                "q": "Anthropic MCP",
                "cursor": ""  # Empty cursor for first page
            }
        },
        # Natural language queries
        {"query": "Search for tweets about crypto marketplace"},
        {"query": "Find recent discussions about Heurist AI"},
        {"query": "What are people saying about Vitalik Buterin?"}
    ]

    results = {}
    for i, test_case in enumerate(test_cases):
        try:
            result = await agent.handle_message(test_case)
            results[f"case_{i + 1}"] = {
                "input": test_case,
                "output": result
            }
            logger.info(f"General search test case {i + 1} completed successfully")
        except Exception as e:
            logger.error(f"Error in general search test case {i + 1}: {e}")
            results[f"case_{i + 1}"] = {
                "input": test_case,
                "error": str(e)
            }

    return results


async def run_agent():
    agent = TwitterInfoAgent()
    try:
        # Test the identifier extraction logic
        logger.info("Testing username/ID extraction...")
        extraction_results = await test_identifier_extraction(agent)

        # Test ID pattern recognition specifically
        logger.info("Testing ID pattern recognition...")
        id_recognition_results = await test_id_pattern_recognition(agent)

        # Test tweet detail functionality
        logger.info("Testing tweet detail functionality...")
        tweet_detail_results = await test_tweet_detail_fetching(agent)

        # Test general search functionality
        logger.info("Testing general search functionality...")
        general_search_results = await test_general_search(agent)

        test_cases = [
            {"query": "Summarise recent updates of @heurist_ai", "limit": 5},
            {"query": "What has @elonmusk been tweeting lately?", "limit": 5},
            {"query": "Get the recent tweets from cz_binance", "limit": 5},
            {"tool": "get_user_tweets", "tool_arguments": {"username": "realdonaldtrump", "limit": 5}},
            # Test with a numeric ID (this is a placeholder ID)
            {"query": "Get tweets from user_id:778764142412984320", "limit": 5},
            # Add test cases for tweet detail
            {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[0]}},
            {"tool": "get_twitter_detail", "tool_arguments": {"tweet_id": TEST_TWEET_IDS[1], "cursor": ""}},
            # Query-based test cases for tweet detail
            {"query": f"Show me the details and replies for tweet {TEST_TWEET_IDS[0]}"},
            {"query": f"Get all information about this tweet: {TEST_TWEET_IDS[1]}"},
            # Add test cases for general search
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
            except Exception as e:
                logger.error(f"Error in test case {i + 1}: {e}")
                api_results[f"case_{i + 1}"] = {"input": test_case, "error": str(e)}

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "username_extraction_tests": extraction_results,
            "id_pattern_recognition": id_recognition_results,
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
