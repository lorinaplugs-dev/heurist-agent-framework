import asyncio
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent.parent))
from mesh.agents.memory_agent import MemoryAgent  # noqa: E402

load_dotenv()


async def run_agent():
    agent = MemoryAgent()

    # Create session context with API key for user identification
    session_context = {"api_key": "xxxxxx-xxxxxxx"}

    try:
        # Natural language query to store conversation
        agent_input_store = {
            "query": "Save our conversation to memory. We discussed about Imagine not having much users.",
            "raw_data_only": False,
            "session_context": session_context,
        }
        agent_output_store = await agent.handle_message(agent_input_store)

        # Direct tool call to store conversation with metadata
        agent_input_store_direct = {
            "tool": "store_conversation",
            "tool_arguments": {
                "content": "User asked about NFT marketplace development. Explained ERC-721 standards, IPFS storage, and smart contract deployment.",
                "metadata": {"platform": "discord", "topic": "NFTs", "sentiment": "educational"},
            },
            "session_context": session_context,
        }
        agent_output_store_direct = await agent.handle_message(agent_input_store_direct)

        # Natural language query to retrieve conversations
        agent_input_retrieve = {
            "query": "What did we talk about in our previous conversations?",
            "raw_data_only": False,
            "session_context": session_context,
        }
        agent_output_retrieve = await agent.handle_message(agent_input_retrieve)

        # Direct tool call to retrieve conversations
        agent_input_retrieve_direct = {
            "tool": "retrieve_conversations",
            "tool_arguments": {"limit": 5},
            "session_context": session_context,
        }
        agent_output_retrieve_direct = await agent.handle_message(agent_input_retrieve_direct)

        # Natural language query with raw data only
        agent_input_raw = {
            "query": "Show me all stored conversations",
            "raw_data_only": False,
            "session_context": session_context,
        }
        agent_output_raw = await agent.handle_message(agent_input_raw)

        script_dir = Path(__file__).parent
        current_file = Path(__file__).stem
        base_filename = f"{current_file}_example"
        output_file = script_dir / f"{base_filename}.yaml"

        yaml_content = {
            "natural_language_store": {"input": agent_input_store, "output": agent_output_store},
            "direct_store_with_metadata": {"input": agent_input_store_direct, "output": agent_output_store_direct},
            "natural_language_retrieve": {"input": agent_input_retrieve, "output": agent_output_retrieve},
            "direct_retrieve": {"input": agent_input_retrieve_direct, "output": agent_output_retrieve_direct},
            "raw_data_query": {"input": agent_input_raw, "output": agent_output_raw},
        }

        with open(output_file, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, allow_unicode=True, sort_keys=False)

        print(f"Results saved to {output_file}")

    finally:
        await agent.cleanup()


if __name__ == "__main__":
    asyncio.run(run_agent())
