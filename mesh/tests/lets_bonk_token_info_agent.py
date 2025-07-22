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

        # Get test token from graduation results
        test_token = None
        if (
            agent_output_direct_graduate.get("data", {}).get("tokens")
            and len(agent_output_direct_graduate["data"]["tokens"]) > 0
        ):
            test_token = agent_output_direct_graduate["data"]["tokens"][0]["token_info"]["mint_address"]
        else:
            test_token = "4TBi66vi32S7J8X1A6eWfaLHYmUXu7CStcEmsJQdpump"

        # Test latest trades
        trades_input = {"query": f"Show me the latest 20 trades for token {test_token}"}
        trades_output = await agent.handle_message(trades_input)

        trades_direct_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": test_token, "limit": 20},
        }
        trades_direct_output = await agent.handle_message(trades_direct_input)

        # Test latest price
        price_input = {"query": f"What's the current price of token {test_token}?"}
        price_output = await agent.handle_message(price_input)

        price_direct_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": test_token},
        }
        price_direct_output = await agent.handle_message(price_direct_input)

        # Test top buyers
        buyers_input = {"query": f"Show me the top 30 buyers of token {test_token}"}
        buyers_output = await agent.handle_message(buyers_input)

        buyers_direct_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": test_token, "limit": 30},
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

        # Test with different trade limits
        trades_10_input = {"query": f"Get latest 10 trades for {test_token}"}
        trades_10_output = await agent.handle_message(trades_10_input)

        trades_10_direct_input = {
            "tool": "query_latest_trades",
            "tool_arguments": {"token_address": test_token, "limit": 10},
        }
        trades_10_direct_output = await agent.handle_message(trades_10_direct_input)

        # Test with different buyer limits
        buyers_50_input = {"query": f"Show me top 50 buyers of {test_token}"}
        buyers_50_output = await agent.handle_message(buyers_50_input)

        buyers_50_direct_input = {
            "tool": "query_top_buyers",
            "tool_arguments": {"token_address": test_token, "limit": 50},
        }
        buyers_50_direct_output = await agent.handle_message(buyers_50_direct_input)

        # Test with raw_data_only flag
        raw_data_input = {
            "query": "Get top 5 tokens about to graduate",
            "raw_data_only": True,
        }
        raw_data_output = await agent.handle_message(raw_data_input)

        # Test another token if available
        test_token_2 = None
        if (
            agent_output_direct_graduate.get("data", {}).get("tokens")
            and len(agent_output_direct_graduate["data"]["tokens"]) > 1
        ):
            test_token_2 = agent_output_direct_graduate["data"]["tokens"][1]["token_info"]["mint_address"]
        else:
            test_token_2 = "98mb39tPFKQJ4Bif8iVg9mYb9wsfPZgpgN1sxoVTpump"

        # Test second token price
        price_2_input = {"query": f"What's the price of {test_token_2}?"}
        price_2_output = await agent.handle_message(price_2_input)

        price_2_direct_input = {
            "tool": "query_latest_price",
            "tool_arguments": {"token_address": test_token_2},
        }
        price_2_direct_output = await agent.handle_message(price_2_direct_input)

        # Test edge cases
        edge_case_input = {
            "tool": "query_about_to_graduate_tokens",
            "tool_arguments": {"limit": 1},
        }
        edge_case_output = await agent.handle_message(edge_case_input)

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
            "latest_trades": {
                "natural_language_query": {"input": trades_input, "output": trades_output},
                "direct_tool_call": {"input": trades_direct_input, "output": trades_direct_output},
            },
            "latest_price": {
                "natural_language_query": {"input": price_input, "output": price_output},
                "direct_tool_call": {"input": price_direct_input, "output": price_direct_output},
            },
            "top_buyers": {
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
            "trades_10_limit": {
                "natural_language_query": {"input": trades_10_input, "output": trades_10_output},
                "direct_tool_call": {"input": trades_10_direct_input, "output": trades_10_direct_output},
            },
            "buyers_50_limit": {
                "natural_language_query": {"input": buyers_50_input, "output": buyers_50_output},
                "direct_tool_call": {"input": buyers_50_direct_input, "output": buyers_50_direct_output},
            },
            "raw_data_test": {
                "input": raw_data_input,
                "output": raw_data_output,
            },
            "second_token_price": {
                "natural_language_query": {"input": price_2_input, "output": price_2_output},
                "direct_tool_call": {"input": price_2_direct_input, "output": price_2_direct_output},
            },
            "edge_case_test": {
                "input": edge_case_input,
                "output": edge_case_output,
            },
            "test_metadata": {
                "test_token_used": test_token,
                "test_token_2_used": test_token_2,
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
