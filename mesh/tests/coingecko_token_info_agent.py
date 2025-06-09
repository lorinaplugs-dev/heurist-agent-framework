import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Common sleep duration (in seconds)
REQUEST_SLEEP = 10

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.coingecko_token_info_agent import CoinGeckoTokenInfoAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = CoinGeckoTokenInfoAgent()
    try:
        # Test with a natural language query
        await asyncio.sleep(REQUEST_SLEEP)
        query_input = {"query": "Get information about MONA"}
        query_result = await agent.handle_message(query_input)
        print(f"Natural Language Query Result: {query_result}")

        # Test direct tool calls for each tool
        # 1. get_token_info
        await asyncio.sleep(REQUEST_SLEEP)
        token_info_input = {
            "tool": "get_token_info",
            "tool_arguments": {"coingecko_id": "bitcoin"},
            "raw_data_only": True,
        }
        token_info_result = await agent.handle_message(token_info_input)
        print(f"Get Token Info Tool Result: {token_info_result}")

        # 2. get_trending_coins
        await asyncio.sleep(REQUEST_SLEEP)
        trending_coins_input = {
            "tool": "get_trending_coins",
            "tool_arguments": {},
            "raw_data_only": True,
        }
        trending_coins_result = await agent.handle_message(trending_coins_input)
        print(f"Get Trending Coins Tool Result: {trending_coins_result}")

        # 3. get_token_price_multi
        await asyncio.sleep(REQUEST_SLEEP)
        price_multi_input = {
            "tool": "get_token_price_multi",
            "tool_arguments": {
                "ids": "bitcoin,ethereum",
                "vs_currencies": "usd",
                "include_market_cap": True,
                "include_24hr_vol": True,
                "include_24hr_change": True,
            },
            "raw_data_only": True,
        }
        price_multi_result = await agent.handle_message(price_multi_input)
        print(f"Get Token Price Multi Tool Result: {price_multi_result}")

        # 4. get_categories_list
        await asyncio.sleep(REQUEST_SLEEP)
        categories_list_input = {
            "tool": "get_categories_list",
            "tool_arguments": {},
            "raw_data_only": True,
        }
        categories_list_result = await agent.handle_message(categories_list_input)
        print(f"Get Categories List Tool Result: {categories_list_result}")

        # 5. get_category_data
        await asyncio.sleep(REQUEST_SLEEP)
        category_data_input = {
            "tool": "get_category_data",
            "tool_arguments": {"order": "market_cap_desc"},
            "raw_data_only": True,
        }
        category_data_result = await agent.handle_message(category_data_input)
        print(f"Get Category Data Tool Result: {category_data_result}")

        # 6. get_tokens_by_category
        await asyncio.sleep(REQUEST_SLEEP)
        tokens_by_category_input = {
            "tool": "get_tokens_by_category",
            "tool_arguments": {
                "category_id": "layer-1",
                "vs_currency": "usd",
                "order": "market_cap_desc",
                "per_page": 10,
                "page": 1,
            },
            "raw_data_only": True,
        }
        tokens_by_category_result = await agent.handle_message(tokens_by_category_input)
        print(f"Get Tokens By Category Tool Result: {tokens_by_category_result}")

        # 7. get_trending_pools
        await asyncio.sleep(REQUEST_SLEEP)
        trending_pools_input = {
            "tool": "get_trending_pools",
            "tool_arguments": {
                "include": "base_token",
                "pools": 4,
            },
            "raw_data_only": False,
        }
        trending_pools_result = await agent.handle_message(trending_pools_input)
        print(f"Get Trending Pools Tool Result: {trending_pools_result}")

        # 8. get_top_token_holders
        await asyncio.sleep(REQUEST_SLEEP)
        top_holders_input = {
            "tool": "get_top_token_holders",
            "tool_arguments": {
                "network": "base",
                "address": "0xEF22cb48B8483dF6152e1423b19dF5553BbD818b",
            },
            "raw_data_only": False,
        }
        top_holders_result = await agent.handle_message(top_holders_input)
        print(f"Get Top Token Holders Tool Result: {top_holders_result}")

        # Test with raw data only for natural language query
        await asyncio.sleep(REQUEST_SLEEP)
        raw_input = {
            "query": "Compare Bitcoin and Ethereum prices",
            "raw_data_only": True,
        }
        raw_result = await agent.handle_message(raw_input)
        print(f"Raw Data Query Result: {raw_result}")

        # Save all test results to a YAML file
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "natural_language_query": {"input": query_input, "output": query_result},
            "token_info_tool": {"input": token_info_input, "output": token_info_result},
            "trending_coins_tool": {"input": trending_coins_input, "output": trending_coins_result},
            "price_multi_tool": {"input": price_multi_input, "output": price_multi_result},
            "categories_list_tool": {"input": categories_list_input, "output": categories_list_result},
            "category_data_tool": {"input": category_data_input, "output": category_data_result},
            "tokens_by_category_tool": {"input": tokens_by_category_input, "output": tokens_by_category_result},
            "trending_pools_tool": {"input": trending_pools_input, "output": trending_pools_result},
            "top_holders_tool": {"input": top_holders_input, "output": top_holders_result},
            "raw_data_query": {"input": raw_input, "output": raw_result},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
