#!/usr/bin/env python3
"""
Black-box test: spawn the server exactly as an MCP client would, speak raw
newline-delimited JSON-RPC over pipes, then check

  * every line on stdout is a valid JSON-RPC 2.0 message  (STDIO isolation)
  * invalid input maps to -32602, unknown tool maps to -32602
  * domain failures are isError results, not protocol errors
  * diagnostics landed on stderr

Run:  python test_stdio_server.py [path/to/server.py]
"""
import json
import subprocess
import sys
from pathlib import Path

SERVER = sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).with_name("server.py"))

LONG_ENOUGH = "Item arrived damaged"


def call(i, name, arguments):
    return {"jsonrpc": "2.0", "id": i, "method": "tools/call", "params": {"name": name, "arguments": arguments}}


# (id, request, expectation) where expectation is ("error", code) | ("ok",) | ("tool_error",)
CASES = [
    (10, call(10, "get_customer_record", {"customer_id": "CUST-00001"}), ("ok",)),
    (11, call(11, "get_customer_record", {"customer_id": "CUST-99999"}), ("tool_error",)),
    (12, call(12, "trigger_refund", {"customer_id": "CUST-00001", "amount": 19.99, "reason": LONG_ENOUGH}), ("ok",)),
    (13, call(13, "trigger_refund", {"customer_id": "CUST-00001", "amount": 5, "reason": LONG_ENOUGH}), ("ok",)),  # int ok
    # ---- must be rejected with -32602 ----
    (20, call(20, "get_customer_record", {"customer_id": "cust-00001"}), ("error", -32602)),
    (21, call(21, "get_customer_record", {"customer_id": "CUST-0001"}), ("error", -32602)),
    (22, call(22, "get_customer_record", {"customer_id": "CUST-000011"}), ("error", -32602)),
    (23, call(23, "get_customer_record", {"customer_id": "CUST-00001\n"}), ("error", -32602)),  # trailing newline
    (24, call(24, "get_customer_record", {"customer_id": "CUST-\u0660\u0660\u0660\u0660\u0661"}), ("error", -32602)),  # arabic digits
    (25, call(25, "get_customer_record", {"customer_id": 12345}), ("error", -32602)),
    (26, call(26, "get_customer_record", {}), ("error", -32602)),
    (27, call(27, "get_customer_record", {"customer_id": "CUST-00001", "extra": 1}), ("error", -32602)),
    (28, call(28, "trigger_refund", {"customer_id": "CUST-00001", "amount": 0, "reason": LONG_ENOUGH}), ("error", -32602)),
    (29, call(29, "trigger_refund", {"customer_id": "CUST-00001", "amount": -3.5, "reason": LONG_ENOUGH}), ("error", -32602)),
    (30, call(30, "trigger_refund", {"customer_id": "CUST-00001", "amount": "19.99", "reason": LONG_ENOUGH}), ("error", -32602)),
    (31, call(31, "trigger_refund", {"customer_id": "CUST-00001", "amount": True, "reason": LONG_ENOUGH}), ("error", -32602)),
    (32, call(32, "trigger_refund", {"customer_id": "CUST-00001", "amount": 1.005, "reason": LONG_ENOUGH}), ("error", -32602)),
    (33, call(33, "trigger_refund", {"customer_id": "CUST-00001", "amount": 1e9, "reason": LONG_ENOUGH}), ("error", -32602)),
    (34, call(34, "trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": "too short"}), ("error", -32602)),
    (35, call(35, "trigger_refund", {"customer_id": "CUST-00001", "amount": 10, "reason": "   short    "}), ("error", -32602)),  # padding doesn't count
    (36, call(36, "trigger_refund", {"customer_id": "CUST-00001", "amount": 10}), ("error", -32602)),
    (37, call(37, "trigger_refund", {"customer_id": "CUST-00001", "amount": None, "reason": LONG_ENOUGH}), ("error", -32602)),
    (38, call(38, "no_such_tool", {}), ("error", -32602)),
]


def main() -> int:
    requests = [
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "stdio-test", "version": "0"},
            },
        },
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        *[c[1] for c in CASES],
    ]
    payload = "".join(json.dumps(r) + "\n" for r in requests)

    proc = subprocess.run(
        [sys.executable, SERVER], input=payload, capture_output=True, text=True, timeout=30
    )

    failures: list[str] = []

    # 1) stdout purity ------------------------------------------------------
    responses: dict[int, dict] = {}
    for n, line in enumerate(proc.stdout.splitlines(), 1):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            failures.append(f"stdout line {n} is not JSON (stdio pollution): {line[:80]!r}")
            continue
        if msg.get("jsonrpc") != "2.0":
            failures.append(f"stdout line {n} is JSON but not JSON-RPC 2.0: {line[:80]!r}")
            continue
        if "id" in msg:
            responses[msg["id"]] = msg

    # 2) protocol flow ------------------------------------------------------
    if 1 not in responses or "result" not in responses[1]:
        failures.append("initialize did not return a result")
    tools = [t["name"] for t in responses.get(2, {}).get("result", {}).get("tools", [])]
    if sorted(tools) != ["get_customer_record", "trigger_refund"]:
        failures.append(f"tools/list returned {tools}")

    # 3) per-case expectations ----------------------------------------------
    for rid, _, expect in CASES:
        resp = responses.get(rid)
        if resp is None:
            failures.append(f"case {rid}: no response")
            continue
        if expect[0] == "error":
            code = resp.get("error", {}).get("code")
            if code != expect[1]:
                failures.append(f"case {rid}: expected JSON-RPC error {expect[1]}, got {resp}")
        elif expect[0] == "ok":
            r = resp.get("result")
            if r is None or r.get("isError"):
                failures.append(f"case {rid}: expected success, got {resp}")
        elif expect[0] == "tool_error":
            r = resp.get("result")
            if r is None or not r.get("isError"):
                failures.append(f"case {rid}: expected isError result, got {resp}")

    # 4) diagnostics went to stderr -----------------------------------------
    if "starting customer-ops" not in proc.stderr:
        failures.append("expected startup log on stderr")

    total = len(CASES) + 3
    if failures:
        print(f"FAILED ({len(failures)} problem(s)):")
        for f in failures:
            print("  -", f)
        print("\n--- stderr ---\n" + proc.stderr)
        return 1
    print(f"OK: {total} checks passed; {len(proc.stdout.splitlines())} stdout lines, all valid JSON-RPC.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
