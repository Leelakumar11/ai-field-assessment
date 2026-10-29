#!/usr/bin/env python3
"""
LLM gateway: POST /v1/generate  {"prompt": "..."}  ->  text/event-stream

    client --> [ gateway ] --> provider.stream(prompt)      (async iterator of text deltas)
    client <-- SSE frames  <-- StreamRedactor <-----------+

Streaming design
  * one asyncio task per connection, pure stdlib (asyncio streams, chunked encoding)
  * every provider delta goes through StreamRedactor.feed(); whatever is safe is
    written immediately and flushed with `await writer.drain()` (that await is also
    the backpressure: a slow client slows the provider read instead of growing a queue)
  * memory per connection is O(1): the redactor's holdback buffer (<700 chars)
  * if the client disconnects, the provider iterator is closed right away so we
    stop paying for tokens nobody will read
  * provider failures mid-stream become a sanitised SSE `error` event; no upstream
    detail is forwarded
  * `Cache-Control: no-cache` + `X-Accel-Buffering: no` so proxies don't buffer
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import AsyncIterator, Callable, Protocol

from pii_redactor import StreamRedactor

log = logging.getLogger("llm-gateway")
MAX_BODY = 256 * 1024
REASONS = {200: "OK", 400: "Bad Request", 404: "Not Found", 405: "Method Not Allowed",
           413: "Payload Too Large", 500: "Internal Server Error"}


class Provider(Protocol):
    def stream(self, prompt: str) -> AsyncIterator[str]: ...


class MockProvider:
    """Stands in for an LLM: emits a canned answer containing PII in tiny, awkward chunks."""

    TEXT = (
        "Sure! You can reach Jane at jane.doe@example.com or john_smith+test@mail.co.uk. "
        "Her SSN is 123-45-6789 and her card is 4111 1111 1111 1111 (order 20240117, total 1,234.56 USD). "
        "Anything else I can help with?"
    )

    def __init__(self, chunk: int = 3, delay_s: float = 0.0, text: str | None = None) -> None:
        self.chunk, self.delay_s, self.text = chunk, delay_s, text or self.TEXT

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        for i in range(0, len(self.text), self.chunk):
            if self.delay_s:
                await asyncio.sleep(self.delay_s)
            else:
                await asyncio.sleep(0)
            yield self.text[i : i + self.chunk]


def sse(event: str | None, payload: str) -> bytes:
    head = f"event: {event}\n" if event else ""
    return f"{head}data: {payload}\n\n".encode()


async def write_chunk(writer: asyncio.StreamWriter, data: bytes) -> None:
    writer.write(b"%x\r\n%s\r\n" % (len(data), data))
    await writer.drain()


class LLMGateway:
    def __init__(self, provider_factory: Callable[[], Provider] | None = None) -> None:
        self._provider_factory = provider_factory or MockProvider

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> asyncio.AbstractServer:
        return await asyncio.start_server(self._handle, host, port)

    # ------------------------------------------------------------------ #
    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            await self._serve(reader, writer)
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # client went away
        except Exception:
            log.exception("unhandled error")
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _respond(self, writer: asyncio.StreamWriter, status: int, body: dict) -> None:
        raw = json.dumps(body).encode()
        writer.write(
            f"HTTP/1.1 {status} {REASONS.get(status, '')}\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(raw)}\r\nConnection: close\r\n\r\n".encode() + raw
        )
        await writer.drain()

    async def _serve(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), 10)
        except (asyncio.LimitOverrunError, asyncio.TimeoutError):
            return await self._respond(writer, 400, {"error": "bad_request"})
        lines = head.decode("latin-1").split("\r\n")
        try:
            method, path, _ = lines[0].split(" ", 2)
        except ValueError:
            return await self._respond(writer, 400, {"error": "bad_request"})
        headers = {k.strip().lower(): v.strip() for k, _, v in (l.partition(":") for l in lines[1:] if ":" in l)}

        try:
            length = int(headers.get("content-length", "0"))
        except ValueError:
            return await self._respond(writer, 400, {"error": "bad_request"})
        if length > MAX_BODY:
            return await self._respond(writer, 413, {"error": "payload_too_large"})
        body = await reader.readexactly(length) if length else b""

        if path != "/v1/generate":
            return await self._respond(writer, 404, {"error": "not_found"})
        if method != "POST":
            return await self._respond(writer, 405, {"error": "method_not_allowed"})
        try:
            prompt = json.loads(body)["prompt"]
            if not isinstance(prompt, str) or not prompt:
                raise ValueError
        except (ValueError, KeyError, TypeError):
            return await self._respond(writer, 400, {"error": "prompt (non-empty string) required"})

        await self._stream(writer, prompt)

    async def _stream(self, writer: asyncio.StreamWriter, prompt: str) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream; charset=utf-8\r\n"
            b"Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        redactor = StreamRedactor()  # per-request state
        source = self._provider_factory().stream(prompt)
        try:
            async for delta in source:
                safe = redactor.feed(delta)
                if safe:
                    await write_chunk(writer, sse(None, json.dumps({"delta": safe}, ensure_ascii=False)))
            tail = redactor.flush()
            if tail:
                await write_chunk(writer, sse(None, json.dumps({"delta": tail}, ensure_ascii=False)))
            await write_chunk(writer, sse(None, "[DONE]"))
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.CancelledError):
            raise  # client gone / shutdown: `finally` stops the upstream
        except Exception:
            log.exception("provider failed mid-stream")  # details stay server-side
            err = json.dumps({"error": {"code": "upstream_error", "message": "The model provider failed."}})
            try:
                await write_chunk(writer, sse("error", err))
                writer.write(b"0\r\n\r\n")
                await writer.drain()
            except ConnectionError:
                pass
        finally:
            aclose = getattr(source, "aclose", None)
            if aclose:
                await aclose()


async def amain(host: str, port: int) -> None:
    server = await LLMGateway(lambda: MockProvider(chunk=4, delay_s=0.02)).start(host, port)
    addr = server.sockets[0].getsockname()
    log.info("listening on http://%s:%d/v1/generate", addr[0], addr[1])
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8082)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    try:
        asyncio.run(amain(a.host, a.port))
    except KeyboardInterrupt:
        pass
