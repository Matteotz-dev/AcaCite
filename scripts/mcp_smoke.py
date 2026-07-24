"""Non-mutating protocol smoke test for the deployed MCP transport."""

import asyncio

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    async with streamable_http_client("http://127.0.0.1:8001/mcp") as streams:
        read, write, _ = streams
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("tools:", ", ".join(tool.name for tool in tools.tools))
            result = await session.call_tool("research_status", {})
            if result.isError:
                raise RuntimeError(result.content)
            print("research_status:", result.structuredContent)


if __name__ == "__main__":
    asyncio.run(main())
