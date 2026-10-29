# AI Field Developer / Solutions Engineer assessment: four solutions

| # | Task | Stack | Verified here |
|---|------|-------|---------------|
| 1 | MCP server, strict validation, stdio isolation | `mcp` SDK + Pydantic v2 | **Not run** (see note) |
| 2 | MCP security gateway (Bearer/JWT role, `admin_*` gating) | stdlib | 18 tests pass |
| 3 | LLM gateway streaming PII guardrail | stdlib asyncio | 15 tests + 120k-run fuzz |
| 4 | Token-aware rate limiter + failover router (on-disk SQLite) | stdlib | 28 tests pass |

> **Task 1 note:** the build sandbox had no network, so `mcp` and `pydantic` could not be installed
> and `server.py` has never been executed. I compile-checked it and validated its black-box test harness
> against a stdlib stub, but please run it once before relying on it:
> `pip install -r requirements.txt && python task1_mcp_server/test_stdio_server.py`
> The raw-handler registration (below) is the part most worth confirming against your installed SDK version.

## Run everything
```bash
pip install -r requirements.txt                 # Task 1 only
python task1_mcp_server/test_stdio_server.py    # spawns the server, speaks raw JSON-RPC
for t in task2_mcp_gateway:test_gateway task3_llm_stream_guardrail:test_guardrail task4_router_fallback:test_router; do
  (cd ${t%%:*} && python -m unittest ${t##*:})
done
```
Manual demos: `python task2_mcp_gateway/mock_downstream.py` then `GATEWAY_JWT_SECRET=s python task2_mcp_gateway/gateway.py`;
`python task3_llm_stream_guardrail/llm_gateway.py` then `curl -N -d '{"prompt":"hi"}' localhost:8082/v1/generate`.

## Design decisions worth discussing

**Task 1**
- Invalid params must be a real JSON-RPC `-32602`. The SDK's `@call_tool()` decorator turns *every* exception into an `isError` result, so the raw `CallToolRequest` handler is registered instead. Domain failures (customer not found) stay `isError` results, because the request was valid.
- stdout purity: logging is bound to stderr at import, and after the SDK captures the real stdout, `sys.stdout` is rebound to stderr so a stray `print()` can't corrupt the stream.
- Strictness: `fullmatch` with `[0-9]` (not `$`/`\d`, which accept `"CUST-00001\n"` and Arabic-Indic digits), strict types (no `"19.99"` strings or `True` as an amount), NaN/inf rejected, at most 2 decimals, `extra="forbid"`, reason length measured after trimming, validation errors never echo input.
- The SDK's default for unexpected exceptions is code `0` plus `str(exc)`. That leaks internals, so it's wrapped to `-32603 Internal error`.

**Task 2**
- Forwards the original request bytes, never a re-serialisation of what it inspected.
- Deny rule is normalised (`strip().casefold()`), batches are rejected (a batch can hide a forbidden call), and a denied *notification* gets 202 with no body but is still never forwarded.
- JWT is verified (HS256, constant-time compare, `exp`, `alg=none` rejected); unknown roles fall back to viewer.
- Limitation: the downstream response is buffered, so SSE/streamed downstream replies would need a streaming proxy.

**Task 3**
- Holdback is *context-aware*, not a fixed N-char window: only the trailing run that could still grow into an email/SSN/card is held, so TTFT costs about one word and memory is bounded (<700 chars) regardless of response size.
- Fuzzing found a real leak in my first version (a Luhn-failing greedy card candidate swallowed a real card: `4111111111111111 1,234.56`). Fixed by Luhn-checking every window of whole digit groups and unioning all spans, so overlaps can only over-redact. The final property test checks streamed output equals whole-text redaction for random chunkings.
- Known tradeoffs: ~10% of random 13-19-digit sequences pass Luhn and will be redacted; undashed 9-digit SSNs and obfuscated PII (`jane [at] x.com`) are not detected. Add patterns in `pii_redactor.py`.
- Backpressure comes from `await drain()`, and a client disconnect closes the provider iterator immediately.

**Task 4**
- One `BEGIN IMMEDIATE` transaction does check-and-reserve, so it is atomic across tasks *and* across processes sharing the DB file (both tested).
- Tokens are reserved from an estimate, reconciled to real usage, and refunded on failure or client cancellation. Single oversize requests return 413 rather than a misleading "retry later".
- The 3000 ms race uses `asyncio.wait` on a task rather than `wait_for`, so a provider that swallows cancellation can't hold the request past the deadline (tested).
- Fails *closed* (503) if the limiter DB breaks; flip in `handle()` if availability matters more than spend control.
- All errors share one shape with fixed messages; upstream text and tracebacks are logged server-side only (tested with planted secrets).
