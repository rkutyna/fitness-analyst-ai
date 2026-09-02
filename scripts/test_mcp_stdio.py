"""Spawn the MCP server as a subprocess and talk to it over stdio like Hermes
will: initialize, list tools, call two tools. Confirms the protocol layer."""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._vault import LOCAL_DB_PATH
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


async def main():
    params = StdioServerParameters(
        command=str(ROOT / ".venv" / "bin" / "python"),
        args=["-m", "health_advisor.mcp_server", "--vault", str(LOCAL_DB_PATH)],
        env=dict(os.environ), cwd=str(ROOT),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await session.list_tools()
            print("TOOLS EXPOSED:", [t.name for t in tools.tools])

            r = await session.call_tool("summarize_metric",
                                        {"metric": "resting_heart_rate", "period": "90d"})
            print("\nsummarize_metric(resting_heart_rate, 90d):")
            print(" ", r.content[0].text[:400])

            r = await session.call_tool("get_latest", {"metric": "step_count"})
            print("\nget_latest(step_count):")
            print(" ", r.content[0].text[:300])
    print("\nSTDIO PROTOCOL OK.")


asyncio.run(main())
