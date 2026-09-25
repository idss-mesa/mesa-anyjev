"""The 25-line elicitation broker copied from mesa-ducklake's ``tests/llm_e2e/harness/mcp_server.py``
(main 7bc143f; the harness is not shipped in the wheel). No top-level ``mcp`` import."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

# (message, requestedSchema) -> chosen content dict, or None to decline.
Chooser = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any] | None]]


class ElicitationBroker:
    def __init__(self) -> None:
        self.chooser: Chooser | None = None
        self.log: list[dict[str, Any]] = []

    async def __call__(self, context: Any, params: Any) -> Any:
        from mcp import types

        message = getattr(params, "message", "")
        schema = getattr(params, "requested_schema", None) or getattr(params, "requestedSchema", {})
        content = await self.chooser(message, schema) if self.chooser else None
        self.log.append({"message": message, "schema": schema, "content": content})
        if content is None:
            return types.ElicitResult(action="decline")
        return types.ElicitResult(action="accept", content=content)


def subprocess_env(catalog_dsn: str, cache_dir: Path) -> dict[str, str]:
    env = {
        k: v
        for k, v in os.environ.items()
        if k.startswith(("MESA_MCP_", "IRODS_")) or k in ("HOME", "USER", "LANG", "PATH")
    }
    env["MESA_MCP_DUCKLAKE__CATALOG_DSN"] = catalog_dsn
    env["MESA_MCP_DUCKLAKE__CACHE_DIR"] = str(cache_dir)
    return env


@asynccontextmanager
async def open_mesa_mcp(
    *, catalog_dsn: str, cache_dir: Path, stderr_log: Path, timeout: float = 300.0
) -> AsyncIterator[tuple[Any, ElicitationBroker]]:
    """Spawn ``mesa-mcp --transport stdio`` with the broker as the elicitation callback."""
    from mcp.client import Client
    from mcp.client.stdio import StdioServerParameters, stdio_client

    params = StdioServerParameters(
        command=os.environ.get("MESA_E2E_MESA_MCP_CMD", "mesa-mcp"),
        args=["--transport", "stdio"],
        env=subprocess_env(catalog_dsn, cache_dir),
    )
    broker = ElicitationBroker()
    with stderr_log.open("a", encoding="utf-8") as errlog:
        async with Client(
            stdio_client(params, errlog=errlog),
            elicitation_callback=broker,
            read_timeout_seconds=timeout,
        ) as client:
            yield client, broker
