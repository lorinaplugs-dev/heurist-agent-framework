import asyncio
import sys
from pathlib import Path

import yaml

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.arbus_ai_agent import ArbusAgent  # noqa: E402


async def run_agent():
    agent = ArbusAgent()
    try:
        # ---------------------
        # TEST CASES
        # ---------------------
        # Test 1: AI Assistant query
        ai_assistant_query_input = {"query": "What is the current market sentiment for bitcoin?"}
        ai_assistant_query_output = await agent.handle_message(ai_assistant_query_input)

        # Test 2: AI Assistant tool call
        ai_assistant_tool_input = {
            "tool": "ask_ai_assistant",
            "tool_arguments": {"query": "Is Ethereum showing bullish trends?", "days": 10},
        }
        ai_assistant_tool_output = await agent.handle_message(ai_assistant_tool_input)

        # Test 3: AI Assistant additional query
        ai_assistant_query_input_2 = {"query": "What are the key price drivers for $eth?"}
        ai_assistant_query_output_2 = await agent.handle_message(ai_assistant_query_input_2)

        # Test 4: AI Assistant additional tool call
        ai_assistant_tool_input_2 = {
            "tool": "ask_ai_assistant",
            "tool_arguments": {"query": "How is $sol performing in the market?", "days": 14},
        }
        ai_assistant_tool_output_2 = await agent.handle_message(ai_assistant_tool_input_2)

        # Test 5: Assistant Summary query
        summary_query_input = {"query": "Provide an analysis of Bitcoin's recent performance"}
        summary_query_output = await agent.handle_message(summary_query_input)

        # Test 6: Assistant Summary tool call
        summary_tool_input = {
            "tool": "assistant_summary",
            "tool_arguments": {"ticker_or_twitterhandle": "heurist_ai", "day_interval": 7},
        }
        summary_tool_output = await agent.handle_message(summary_tool_input)

        # Test 7: Assistant Summary additional query
        summary_query_input_2 = {"query": "What is the outlook for bitcoin's ecosystem of a week?"}
        summary_query_output_2 = await agent.handle_message(summary_query_input_2)

        # Test 8: Assistant Summary additional tool call
        summary_tool_input_2 = {
            "tool": "assistant_summary",
            "tool_arguments": {"ticker_or_twitterhandle": "mona_witchy", "day_interval": 1},
        }
        summary_tool_output_2 = await agent.handle_message(summary_tool_input_2)

        # Test 9: Report query
        report_query_input = {"query": "Generate a report on Ethereum project developments"}
        report_query_output = await agent.handle_message(report_query_input)

        # Test 10: Report tool call
        report_tool_input = {
            "tool": "report",
            "tool_arguments": {"twitter_handle": "@bitcoin", "category": "projects"},
            "raw_data_only": False,
        }
        report_tool_output = await agent.handle_message(report_tool_input)

        # Test 11: Report additional query
        report_query_input_2 = {"query": "Provide a report on heurist's recent partnerships"}
        report_query_output_2 = await agent.handle_message(report_query_input_2)

        # Test 12: Report additional tool call
        report_tool_input_2 = {
            "tool": "report",
            "tool_arguments": {"twitter_handle": "SpaceX", "category": "projects"},
            "raw_data_only": True,
        }
        report_tool_output_2 = await agent.handle_message(report_tool_input_2)

        # ---------------------
        # SAVE RESULTS
        # ---------------------
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        output_file = script_dir / f"{current_file}_example.yaml"

        yaml_content = {
            "ai_assistant_query_test": {"input": ai_assistant_query_input, "output": ai_assistant_query_output},
            "ai_assistant_tool_test": {"input": ai_assistant_tool_input, "output": ai_assistant_tool_output},
            "ai_assistant_query_test_2": {"input": ai_assistant_query_input_2, "output": ai_assistant_query_output_2},
            "ai_assistant_tool_test_2": {"input": ai_assistant_tool_input_2, "output": ai_assistant_tool_output_2},
            "summary_query_test": {"input": summary_query_input, "output": summary_query_output},
            "summary_tool_test": {"input": summary_tool_input, "output": summary_tool_output},
            "summary_query_test_2": {"input": summary_query_input_2, "output": summary_query_output_2},
            "summary_tool_test_2": {"input": summary_tool_input_2, "output": summary_tool_output_2},
            "report_query_test": {"input": report_query_input, "output": report_query_output},
            "report_tool_test": {"input": report_tool_input, "output": report_tool_output},
            "report_query_test_2": {"input": report_query_input_2, "output": report_query_output_2},
            "report_tool_test_2": {"input": report_tool_input_2, "output": report_tool_output_2},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
