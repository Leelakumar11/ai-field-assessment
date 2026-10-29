#!/usr/bin/env python3
"""
customer-ops MCP server (stdio transport).

Tools
  get_customer_record(customer_id: "CUST-XXXXX")
  trigger_refund(customer_id, amount > 0, reason >= 10 chars)

Design notes (what a reviewer should look for)
  1. STDIO ISOLATION  - stdout carries JSON-RPC and nothing else.
       * logging is configured to stderr before anything else happens
       * after the SDK has bound to the real stdout, `sys.stdout` is rebound to
         stderr so an accidental print() in this process (or a dependency) can
         never corrupt the protocol stream
  2. PROTOCOL ERRORS  - invalid params are returned as real JSON-RPC errors
       (-32602), NOT as `isError: true` tool results. The high-level `@call_tool`
       decorator converts every exception into an isError result, so we register
       the raw CallToolRequest handler instead, which lets McpError propagate to
       the wire as a JSON-RPC error object.
       Domain failures ("customer not found") are a different thing: the request
       was well-formed and the tool ran, so they are `isError: true` results.
  3. VALIDATION       - Pydantic v2, strict mode, extra fields forbidden.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sys
import uuid
from decimal import Decimal
from typing import Annotated, Any

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    ValidationError,
)

import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.shared.exceptions import McpError

# --------------------------------------------------------------------------- #
# Logging: stderr only. Configured at import time, before anything can log.
# --------------------------------------------------------------------------- #
logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [customer-ops] %(message)s",
)
log = logging.getLogger("customer-ops")

# --------------------------------------------------------------------------- #
# Input schemas
# --------------------------------------------------------------------------- #
CUSTOMER_ID_PATTERN = r"^CUST-[0-9]{5}$"
# `[0-9]` (not `\d`, which matches non-ASCII digits) and fullmatch (not `$`,
# which in Python's `re` also matches before a trailing "\n").
_CUSTOMER_ID_RE = re.compile(r"CUST-[0-9]{5}")

MAX_REFUND = 10_000.00  # business sanity cap; adjust to policy


def _check_customer_id(v: str) -> str:
    if not _CUSTOMER_ID_RE.fullmatch(v):
        raise ValueError("must match CUST-XXXXX where X is a digit")
    return v


def _check_amount(v: float) -> float:
    # at most 2 decimal places (currency); Decimal(str(x)) avoids binary-float noise
    if Decimal(str(v)).as_tuple().exponent < -2:  # type: ignore[operator]
        raise ValueError("must have at most 2 decimal places")
    return v


CustomerId = Annotated[
    str,
    Field(strict=True, pattern=CUSTOMER_ID_PATTERN, description="Customer id, format CUST-XXXXX"),
    AfterValidator(_check_customer_id),  # authoritative check; `pattern` is for the advertised schema
]

Amount = Annotated[
    float,
    Field(strict=True, gt=0, le=MAX_REFUND, allow_inf_nan=False, description="Refund amount, > 0"),
    AfterValidator(_check_amount),
]

Reason = Annotated[
    str,
    StringConstraints(strict=True, strip_whitespace=True, min_length=10, max_length=500),
    Field(description="Why the refund is issued (min 10 characters after trimming)"),
]


class GetCustomerRecordArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: CustomerId


class TriggerRefundArgs(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    customer_id: CustomerId
    amount: Amount
    reason: Reason


TOOLS: dict[str, tuple[type[BaseModel], str, types.ToolAnnotations]] = {
    "get_customer_record": (
        GetCustomerRecordArgs,
        "Fetch a customer's record by id (format CUST-XXXXX).",
        types.ToolAnnotations(readOnlyHint=True, idempotentHint=True, openWorldHint=False),
    ),
    "trigger_refund": (
        TriggerRefundArgs,
        "Issue a refund to a customer. Amount must be > 0 and a reason of at least 10 characters is required.",
        types.ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=False, openWorldHint=False),
    ),
}

# --------------------------------------------------------------------------- #
# Fake backing data (replace with a real service)
# --------------------------------------------------------------------------- #
CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST-00001": {"name": "Ada Lovelace", "email": "ada@example.com", "tier": "gold", "balance": 120.50},
    "CUST-00002": {"name": "Alan Turing", "email": "alan@example.com", "tier": "silver", "balance": 0.0},
}
REFUNDS: list[dict[str, Any]] = []

# --------------------------------------------------------------------------- #
# Server
# --------------------------------------------------------------------------- #
server = Server("customer-ops")


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name=name,
            description=desc,
            inputSchema=model.model_json_schema(),
            annotations=ann,
        )
        for name, (model, desc, ann) in TOOLS.items()
    ]


def _rpc_error(code: int, message: str, data: Any = None) -> McpError:
    return McpError(types.ErrorData(code=code, message=message, data=data))


def _validation_error(exc: ValidationError) -> McpError:
    # include_input=False: never echo caller-supplied values back (could be PII/huge)
    details = [
        {"field": ".".join(str(p) for p in e["loc"]) or "(root)", "message": e["msg"]}
        for e in exc.errors(include_url=False, include_context=False, include_input=False)
    ]
    return _rpc_error(types.INVALID_PARAMS, "Invalid params", {"errors": details})


def _text_result(payload: dict[str, Any], *, is_error: bool = False) -> types.ServerResult:
    return types.ServerResult(
        types.CallToolResult(
            content=[types.TextContent(type="text", text=json.dumps(payload))],
            isError=is_error,
        )
    )


def _get_customer_record(args: GetCustomerRecordArgs) -> types.ServerResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _text_result({"error": f"Customer {args.customer_id} not found"}, is_error=True)
    return _text_result({"customer_id": args.customer_id, **record})


def _trigger_refund(args: TriggerRefundArgs) -> types.ServerResult:
    record = CUSTOMERS.get(args.customer_id)
    if record is None:
        return _text_result({"error": f"Customer {args.customer_id} not found"}, is_error=True)
    refund = {
        "refund_id": f"RF-{uuid.uuid4().hex[:12]}",
        "customer_id": args.customer_id,
        "amount": args.amount,
        "reason": args.reason,
        "status": "queued",
    }
    REFUNDS.append(refund)
    log.info("refund queued id=%s customer=%s amount=%.2f", refund["refund_id"], args.customer_id, args.amount)
    return _text_result(refund)


async def handle_call_tool(req: types.CallToolRequest) -> types.ServerResult:
    name = req.params.name
    entry = TOOLS.get(name)
    if entry is None:
        # MCP spec: unknown tool => -32602 Invalid params
        raise _rpc_error(types.INVALID_PARAMS, f"Unknown tool: {name}")

    model = entry[0]
    try:
        args = model.model_validate(req.params.arguments or {})
    except ValidationError as exc:
        log.info("rejected call to %s: %d validation error(s)", name, exc.error_count())
        raise _validation_error(exc) from None

    try:
        if isinstance(args, GetCustomerRecordArgs):
            return _get_customer_record(args)
        return _trigger_refund(args)  # type: ignore[arg-type]
    except McpError:
        raise
    except Exception:
        # The SDK would otherwise send str(exc) with code 0 -> leaks internals.
        log.exception("unexpected failure in tool %s", name)
        raise _rpc_error(types.INTERNAL_ERROR, "Internal error") from None


# Register the *raw* handler (see module docstring, point 2).
server.request_handlers[types.CallToolRequest] = handle_call_tool


async def amain() -> None:
    log.info("starting customer-ops MCP server on stdio")
    async with stdio_server() as (read_stream, write_stream):
        # The SDK has captured the real stdout by now. From here on, anything
        # that writes to sys.stdout goes to stderr instead of the protocol pipe.
        sys.stdout = sys.stderr
        await server.run(read_stream, write_stream, server.create_initialization_options())
    log.info("stdin closed, shutting down")


if __name__ == "__main__":
    asyncio.run(amain())
