import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.arbus_agent import ArbusAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = ArbusAgent()
    try:
        # Test 1: AI Assistant - Direct tool call
        agent_input_ai_direct = {
            "tool": "ask_ai_assistant",
            "tool_arguments": {"query": "What's happening with DeFi markets?", "days": 7},
        }
        agent_output_ai_direct = await agent.handle_message(agent_input_ai_direct)

        # Test 2: AI Assistant - Natural language query
        agent_input_ai_query = {"query": "Is Bitcoin bullish right now?"}
        agent_output_ai_query = await agent.handle_message(agent_input_ai_query)

        # Test 3: Project Analysis - Direct tool call
        agent_input_project_direct = {
            "tool": "analyze_project",
            "tool_arguments": {
                "ticker_or_twitterhandle": "ETH",  # Using ticker symbol
                "day_interval": 3,  # Shorter period to avoid timeouts
            },
        }
        agent_output_project_direct = await agent.handle_message(agent_input_project_direct)

        # Test 4: Project Analysis - Natural language query
        agent_input_project_query = {
            "query": "Analyze Bitcoin's recent developments"  # Changed to Bitcoin
        }
        agent_output_project_query = await agent.handle_message(agent_input_project_query)

        # Test 5: Report Generation - Direct tool call
        agent_input_report_direct = {
            "tool": "generate_report",
            "tool_arguments": {"twitter_handle": "ethereum", "category": "projects"},
        }
        agent_output_report_direct = await agent.handle_message(agent_input_report_direct)

        # Test 6: Report Generation - Natural language query
        agent_input_report_query = {"query": "Generate a report on Ethereum's partnerships"}
        agent_output_report_query = await agent.handle_message(agent_input_report_query)

        # Save results to YAML
        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "ai_assistant_direct": {"input": agent_input_ai_direct, "output": agent_output_ai_direct},
            "ai_assistant_query": {"input": agent_input_ai_query, "output": agent_output_ai_query},
            "project_analysis_direct": {"input": agent_input_project_direct, "output": agent_output_project_direct},
            "project_analysis_query": {"input": agent_input_project_query, "output": agent_output_project_query},
            "report_generation_direct": {"input": agent_input_report_direct, "output": agent_output_report_direct},
            "report_generation_query": {"input": agent_input_report_query, "output": agent_output_report_query},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
