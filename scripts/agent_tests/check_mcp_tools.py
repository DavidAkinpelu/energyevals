#!/usr/bin/env python
"""Check MCP tool awareness and agent tool list inclusion."""
import asyncio

from dotenv import load_dotenv

from energbench.mcp import create_mcp_client
from energbench.tools import create_default_registry


async def main() -> None:
    load_dotenv()

    std_registry = create_default_registry()
    std_tool_names = {tool.name for tool in std_registry.get_all_tools()}

    mcp_client = await create_mcp_client()
    try:
        mcp_tools = mcp_client.list_tools()
        mcp_tool_names = {tool.name for tool in mcp_tools}

        agent_tools = std_registry.get_all_tools() + mcp_tools
        agent_tool_names = {tool.name for tool in agent_tools}

        print("MCP tools:")
        for name in sorted(mcp_tool_names):
            print(f"  - {name}")

        missing = mcp_tool_names - agent_tool_names
        print("\nMissing from agent tool list:", sorted(missing) if missing else "none")
        print("Overlap with standard tools:", sorted(std_tool_names & mcp_tool_names) or "none")
    finally:
        await mcp_client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
