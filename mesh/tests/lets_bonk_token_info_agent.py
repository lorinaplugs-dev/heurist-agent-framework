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

        # Test latest trades
        print("Testing latest trades (query)")
        trades_query_input = {
            "query": "Show me the latest 10 trades for token So11111111111111111111111111111111111111112"
        }
        trades_query_output = await agent.handle_message(trades_query_input)

        print("Testing latest trades (tool)")
        trades_tool_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v", "limit": 25},
        }
        trades_tool_output = await agent.handle_message(trades_tool_input)

        # Test latest price
        print("Testing latest price (query)")
        price_query_input = {"query": "What's the current price of token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263?"}
        price_query_output = await agent.handle_message(price_query_input)

        print("Testing latest price (tool)")
        price_tool_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"},
        }
        price_tool_output = await agent.handle_message(price_tool_input)

        # Test top buyers
        print("Testing top buyers (query)")
        buyers_query_input = {"query": "Show me the top 10 buyers of token MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac"}
        buyers_query_output = await agent.handle_message(buyers_query_input)

        print("Testing top buyers (tool)")
        buyers_tool_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "limit": 40},
        }
        buyers_tool_output = await agent.handle_message(buyers_tool_input)

        # Test top sellers
        print("Testing top sellers (query)")
        sellers_query_input = {
            "query": "Show me the top 10 sellers of token orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE"
        }
        sellers_query_output = await agent.handle_message(sellers_query_input)

        print("Testing top sellers (tool)")
        sellers_tool_input = {
            "tool": "query_top_sellers",
            "tool_arguments": {"token_address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt", "limit": 35},
        }
        sellers_tool_output = await agent.handle_message(sellers_tool_input)

        # Test OHLCV data
        print("Testing OHLCV data (query)")
        ohlcv_query_input = {"query": "Get OHLCV data for token Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"}
        ohlcv_query_output = await agent.handle_message(ohlcv_query_input)

        print("Testing OHLCV data (tool)")
        ohlcv_tool_input = {
            "tool": "query_ohlcv_data",
            "tool_arguments": {"token_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "limit": 50},
        }
        ohlcv_tool_output = await agent.handle_message(ohlcv_tool_input)

        # Test pair address
        print("Testing pair address (query)")
        pair_query_input = {"query": "Get pair address for token So11111111111111111111111111111111111111112"}
        pair_query_output = await agent.handle_message(pair_query_input)

        print("Testing pair address (tool)")
        pair_tool_input = {
            "tool": "query_pair_address",
            "tool_arguments": {"token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        }
        pair_tool_output = await agent.handle_message(pair_tool_input)

        # Test liquidity
        print("Testing liquidity (query)")
        liquidity_query_input = {"query": "Get liquidity for pool address 58oQChx4yWmvKdwLLZzBi4ChoCc2fqCUWBkwMihLYQo2"}
        liquidity_query_output = await agent.handle_message(liquidity_query_input)

        print("Testing liquidity (tool)")
        liquidity_tool_input = {
            "tool": "query_liquidity",
            "tool_arguments": {"pool_address": "9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM"},
        }
        liquidity_tool_output = await agent.handle_message(liquidity_tool_input)

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
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
