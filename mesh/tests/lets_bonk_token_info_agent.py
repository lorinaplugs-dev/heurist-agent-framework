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
        agent_input = {"query": "Show me top 10 tokens about to graduate on LetsBonk.fun"}
        agent_output = await agent.handle_message(agent_input)

        agent_input_direct_graduate = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 10},
        }
        agent_output_direct_graduate = await agent.handle_message(agent_input_direct_graduate)

        # Test latest trades with a known Solana token address
        trades_input = {"query": "Show me the latest 20 trades for token So11111111111111111111111111111111111111112"}
        trades_output = await agent.handle_message(trades_input)

        trades_direct_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": "So11111111111111111111111111111111111111112", "limit": 20},
        }
        trades_direct_output = await agent.handle_message(trades_direct_input)

        # Test latest price with USDC token address
        price_input = {"query": "What's the current price of token EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v?"}
        price_output = await agent.handle_message(price_input)

        price_direct_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"},
        }
        price_direct_output = await agent.handle_message(price_direct_input)

        # Test top buyers with another authentic Solana token (BONK)
        buyers_input = {"query": "Show me the top 30 buyers of token DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263"}
        buyers_output = await agent.handle_message(buyers_input)

        buyers_direct_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263", "limit": 30},
        }
        buyers_direct_output = await agent.handle_message(buyers_direct_input)

        # Test with different limits for graduation tokens
        graduate_50_input = {"query": "Get top 50 tokens about to graduate on LetsBonk.fun"}
        graduate_50_output = await agent.handle_message(graduate_50_input)

        graduate_50_direct_input = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 50},
        }
        graduate_50_direct_output = await agent.handle_message(graduate_50_direct_input)

        # Test with custom date for graduation tokens
        graduate_date_input = {"query": "Show me tokens about to graduate since last week"}
        graduate_date_output = await agent.handle_message(graduate_date_input)

        graduate_date_direct_input = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 25, "since_date": "2025-07-15T00:00:00Z"},
        }
        graduate_date_direct_output = await agent.handle_message(graduate_date_direct_input)

        # Test with different trade limits using Raydium token
        trades_10_input = {"query": "Get latest 10 trades for 4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R"}
        trades_10_output = await agent.handle_message(trades_10_input)

        trades_10_direct_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R", "limit": 10},
        }
        trades_10_direct_output = await agent.handle_message(trades_10_direct_input)

        # Test with different buyer limits using Serum token
        buyers_50_input = {"query": "Show me top 50 buyers of SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt"}
        buyers_50_output = await agent.handle_message(buyers_50_input)

        buyers_50_direct_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt", "limit": 50},
        }
        buyers_50_direct_output = await agent.handle_message(buyers_50_direct_input)

        # Test with raw_data_only flag
        raw_data_input = {
            "query": "Get top 5 tokens about to graduate",
            "raw_data_only": True,
        }
        raw_data_output = await agent.handle_message(raw_data_input)

        # Test with Mango token price
        price_2_input = {"query": "What's the price of MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac?"}
        price_2_output = await agent.handle_message(price_2_input)

        price_2_direct_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac"},
        }
        price_2_direct_output = await agent.handle_message(price_2_direct_input)

        # Test edge cases with minimal limit
        edge_case_input = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 1},
        }
        edge_case_output = await agent.handle_message(edge_case_input)

        # Test with another authentic token for trades (Jupiter)
        jupiter_trades_input = {"query": "Show me latest 15 trades for JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN"}
        jupiter_trades_output = await agent.handle_message(jupiter_trades_input)

        jupiter_trades_direct_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN", "limit": 15},
        }
        jupiter_trades_direct_output = await agent.handle_message(jupiter_trades_direct_input)

        # Test with USDT price check
        usdt_price_input = {"query": "What's the current price of Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB?"}
        usdt_price_output = await agent.handle_message(usdt_price_input)

        usdt_price_direct_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"},
        }
        usdt_price_direct_output = await agent.handle_message(usdt_price_direct_input)

        # Test top buyers for Orca token
        orca_buyers_input = {"query": "Show me top 25 buyers of orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE"}
        orca_buyers_output = await agent.handle_message(orca_buyers_input)

        orca_buyers_direct_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE", "limit": 25},
        }
        orca_buyers_direct_output = await agent.handle_message(orca_buyers_direct_input)

        # Save results to YAML file
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "about_to_graduate": {
                "natural_language_query": {"input": agent_input, "output": agent_output},
                "direct_tool_call": {"input": agent_input_direct_graduate, "output": agent_output_direct_graduate},
            },
            "latest_trades_sol": {
                "natural_language_query": {"input": trades_input, "output": trades_output},
                "direct_tool_call": {"input": trades_direct_input, "output": trades_direct_output},
            },
            "latest_price_usdc": {
                "natural_language_query": {"input": price_input, "output": price_output},
                "direct_tool_call": {"input": price_direct_input, "output": price_direct_output},
            },
            "top_buyers_bonk": {
                "natural_language_query": {"input": buyers_input, "output": buyers_output},
                "direct_tool_call": {"input": buyers_direct_input, "output": buyers_direct_output},
            },
            "graduation_50_limit": {
                "natural_language_query": {"input": graduate_50_input, "output": graduate_50_output},
                "direct_tool_call": {"input": graduate_50_direct_input, "output": graduate_50_direct_output},
            },
            "graduation_custom_date": {
                "natural_language_query": {"input": graduate_date_input, "output": graduate_date_output},
                "direct_tool_call": {"input": graduate_date_direct_input, "output": graduate_date_direct_output},
            },
            "trades_10_limit_raydium": {
                "natural_language_query": {"input": trades_10_input, "output": trades_10_output},
                "direct_tool_call": {"input": trades_10_direct_input, "output": trades_10_direct_output},
            },
            "buyers_50_limit_serum": {
                "natural_language_query": {"input": buyers_50_input, "output": buyers_50_output},
                "direct_tool_call": {"input": buyers_50_direct_input, "output": buyers_50_direct_output},
            },
            "raw_data_test": {
                "input": raw_data_input,
                "output": raw_data_output,
            },
            "mango_token_price": {
                "natural_language_query": {"input": price_2_input, "output": price_2_output},
                "direct_tool_call": {"input": price_2_direct_input, "output": price_2_direct_output},
            },
            "jupiter_trades": {
                "natural_language_query": {"input": jupiter_trades_input, "output": jupiter_trades_output},
                "direct_tool_call": {"input": jupiter_trades_direct_input, "output": jupiter_trades_direct_output},
            },
            "usdt_price": {
                "natural_language_query": {"input": usdt_price_input, "output": usdt_price_output},
                "direct_tool_call": {"input": usdt_price_direct_input, "output": usdt_price_direct_output},
            },
            "orca_buyers": {
                "natural_language_query": {"input": orca_buyers_input, "output": orca_buyers_output},
                "direct_tool_call": {"input": orca_buyers_direct_input, "output": orca_buyers_direct_output},
            },
            "edge_case_test": {
                "input": edge_case_input,
                "output": edge_case_output,
            },
            "test_metadata": {
                "tokens_tested": {
                    "sol_wrapped": "So11111111111111111111111111111111111111112",
                    "usdc": "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    "bonk": "DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263",
                    "raydium": "4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R",
                    "serum": "SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt",
                    "mango": "MangoCzJ36AjZyKwVj3VnYU4GTonjfVEnJmvvWaxLac",
                    "jupiter": "JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN",
                    "usdt": "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                    "orca": "orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE",
                },
                "agent_name": "LetsBonkTokenInfoAgent",
            },
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
