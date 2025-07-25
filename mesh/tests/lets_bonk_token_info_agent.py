import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.lets_bonk_token_info_agent import LetsBonkTokenInfoAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = LetsBonkTokenInfoAgent()
    try:
        # Test about to graduate tokens
        print("Testing about to graduate tokens (query)")
        graduate_query_input = {"query": "Show me top 10 tokens about to graduate on LetsBonk.fun"}
        graduate_query_output = await agent.handle_message(graduate_query_input)

        print("Testing about to graduate tokens (tool)")
        graduate_tool_input = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 15},
        }
        graduate_tool_output = await agent.handle_message(graduate_tool_input)

        # Test latest trades - using WSOL which has active trading
        print("Testing latest trades (query)")
        trades_query_input = {
            "query": "Show me the latest 10 trades for token So11111111111111111111111111111111111111112"
        }
        trades_query_output = await agent.handle_message(trades_query_input)

        print("Testing latest trades (tool) - Using active LetsBonk token")
        trades_tool_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {
                "token_address": "AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk",
                "limit": 25,
            },  # porkfolio token
        }
        trades_tool_output = await agent.handle_message(trades_tool_input)

        # Test latest price - using tokens from the graduated list
        print("Testing latest price (query)")
        price_query_input = {
            "query": "What's the current price of token AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk?"
        }  # porkfolio
        price_query_output = await agent.handle_message(price_query_input)

        print("Testing latest price (tool)")
        price_tool_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": "GN7BPjVW6UfexZ1Tu6UTa9X7Qd9pJBDNEstR5Lv3bonk"},  # Groktor token
        }
        price_tool_output = await agent.handle_message(price_tool_input)

        # Test top buyers - using tokens that actually have trading data
        print("Testing top buyers (query)")
        buyers_query_input = {
            "query": "Show me the top 10 buyers of token AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk"
        }  # porkfolio
        buyers_query_output = await agent.handle_message(buyers_query_input)

        print("Testing top buyers (tool)")
        buyers_tool_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {
                "token_address": "F9WhPkcmLCVfgKucysxUWbqjrZfUYFsyQkxYnam9bonk",
                "limit": 40,
            },  # "How is this not tokenized yet"
        }
        buyers_tool_output = await agent.handle_message(buyers_tool_input)

        # Test top sellers - using active LetsBonk tokens
        print("Testing top sellers (query)")
        sellers_query_input = {
            "query": "Show me the top 10 sellers of token 6SuHwUtzC1yZQhrfY3GZqcphPfhG2k9rPeBbB9Q3bonk"  # GROKPHONE
        }
        sellers_query_output = await agent.handle_message(sellers_query_input)

        print("Testing top sellers (tool)")
        sellers_tool_input = {
            "tool": "query_top_sellers",
            "tool_arguments": {
                "token_address": "AKmQ3Uv7yZzU6YgTGf7hXfETcJu8kj6CaqvWmiv7bonk",
                "limit": 35,
            },  # gecko.jpg
        }
        sellers_tool_output = await agent.handle_message(sellers_tool_input)

        # Test OHLCV data - using active tokens
        print("Testing OHLCV data (query)")
        ohlcv_query_input = {
            "query": "Get OHLCV data for token AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk"
        }  # porkfolio
        ohlcv_query_output = await agent.handle_message(ohlcv_query_input)

        print("Testing OHLCV data (tool)")
        ohlcv_tool_input = {
            "tool": "query_ohlcv_data",
            "tool_arguments": {
                "token_address": "E8XPu39wNY4HfRgCRMmp2vee75N9gCAd9PnPsoesbonk",
                "limit": 50,
            },  # UMS token
        }
        ohlcv_tool_output = await agent.handle_message(ohlcv_tool_input)

        # Test pair address - using WSOL which has many pairs
        print("Testing pair address (query)")
        pair_query_input = {"query": "Get pair address for token So11111111111111111111111111111111111111112"}
        pair_query_output = await agent.handle_message(pair_query_input)

        print("Testing pair address (tool)")
        pair_tool_input = {
            "tool": "query_pair_address",
            "tool_arguments": {"token_address": "AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk"},  # porkfolio
        }
        pair_tool_output = await agent.handle_message(pair_tool_input)

        # Test liquidity - using the market address from porkfolio
        print("Testing liquidity (query)")
        liquidity_query_input = {
            "query": "Get liquidity for pool address EUb3rQrPBdEZdTo8i6HtxHTMxtfKxBnGmqmAQxcXgSk4"
        }  # porkfolio market
        liquidity_query_output = await agent.handle_message(liquidity_query_input)

        print("Testing liquidity (tool)")
        liquidity_tool_input = {
            "tool": "query_liquidity",
            "tool_arguments": {
                "pool_address": "EJdLYGBMt6uvyijmGgGjz6aCMD4hkGtt6Tk5iV9YnH9b"
            },  # "How is this not tokenized yet" market
        }
        liquidity_tool_output = await agent.handle_message(liquidity_tool_input)

        # Test recently created tokens
        print("Testing recently created tokens (query)")
        created_query_input = {"query": "Show me recently created LetsBonk.fun tokens"}
        created_query_output = await agent.handle_message(created_query_input)

        print("Testing recently created tokens (tool)")
        created_tool_input = {
            "tool": "query_recently_created_tokens",
            "tool_arguments": {"limit": 20},
        }
        created_tool_output = await agent.handle_message(created_tool_input)

        # Test bonding curve progress - using active LetsBonk tokens
        print("Testing bonding curve progress (query)")
        bonding_query_input = {
            "query": "Calculate bonding curve progress for token AF5ZJKsC12VsvmLASF6JWDZQjeKMBdD7mCQYSHHnbonk"  # porkfolio
        }
        bonding_query_output = await agent.handle_message(bonding_query_input)

        print("Testing bonding curve progress (tool)")
        bonding_tool_input = {
            "tool": "query_bonding_curve_progress",
            "tool_arguments": {
                "token_address": "F9WhPkcmLCVfgKucysxUWbqjrZfUYFsyQkxYnam9bonk"
            },  # "How is this not tokenized yet"
        }
        bonding_tool_output = await agent.handle_message(bonding_tool_input)

        # Test tokens above 95%
        print("Testing tokens above 95% bonding curve progress (query)")
        percent_95_query_input = {"query": "Show me tokens above 95% bonding curve progress"}
        percent_95_query_output = await agent.handle_message(percent_95_query_input)

        print("Testing tokens above 95% bonding curve progress (tool)")
        percent_95_tool_input = {
            "tool": "query_tokens_above_95_percent",
            "tool_arguments": {"limit": 15},
        }
        percent_95_tool_output = await agent.handle_message(percent_95_tool_input)

        # Save results to YAML file
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "about_to_graduate": {
                "natural_language_query": {"input": graduate_query_input, "output": graduate_query_output},
                "direct_tool_call": {"input": graduate_tool_input, "output": graduate_tool_output},
            },
            "latest_trades": {
                "natural_language_query": {"input": trades_query_input, "output": trades_query_output},
                "direct_tool_call": {"input": trades_tool_input, "output": trades_tool_output},
            },
            "latest_price": {
                "natural_language_query": {"input": price_query_input, "output": price_query_output},
                "direct_tool_call": {"input": price_tool_input, "output": price_tool_output},
            },
            "top_buyers": {
                "natural_language_query": {"input": buyers_query_input, "output": buyers_query_output},
                "direct_tool_call": {"input": buyers_tool_input, "output": buyers_tool_output},
            },
            "top_sellers": {
                "natural_language_query": {"input": sellers_query_input, "output": sellers_query_output},
                "direct_tool_call": {"input": sellers_tool_input, "output": sellers_tool_output},
            },
            "ohlcv_data": {
                "natural_language_query": {"input": ohlcv_query_input, "output": ohlcv_query_output},
                "direct_tool_call": {"input": ohlcv_tool_input, "output": ohlcv_tool_output},
            },
            "pair_address": {
                "natural_language_query": {"input": pair_query_input, "output": pair_query_output},
                "direct_tool_call": {"input": pair_tool_input, "output": pair_tool_output},
            },
            "liquidity": {
                "natural_language_query": {"input": liquidity_query_input, "output": liquidity_query_output},
                "direct_tool_call": {"input": liquidity_tool_input, "output": liquidity_tool_output},
            },
            "recently_created_tokens": {
                "natural_language_query": {"input": created_query_input, "output": created_query_output},
                "direct_tool_call": {"input": created_tool_input, "output": created_tool_output},
            },
            "bonding_curve_progress": {
                "natural_language_query": {"input": bonding_query_input, "output": bonding_query_output},
                "direct_tool_call": {"input": bonding_tool_input, "output": bonding_tool_output},
            },
            "tokens_above_95_percent": {
                "natural_language_query": {"input": percent_95_query_input, "output": percent_95_query_output},
                "direct_tool_call": {"input": percent_95_tool_input, "output": percent_95_tool_output},
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
