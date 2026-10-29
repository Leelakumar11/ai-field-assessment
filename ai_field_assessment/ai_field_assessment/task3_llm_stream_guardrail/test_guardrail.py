import asyncio
import json
import random
import re
import time
import unittest

import llm_gateway
from pii_redactor import MAX_EMAIL_LEN, REDACTION, StreamRedactor, redact_stream, redact_text


def stream_all(text, sizes):
    r, out, i = StreamRedactor(), [], 0
    while i < len(text):
        k = sizes(); out.append(r.feed(text[i:i + k])); i += k
    out.append(r.flush())
    return "".join(out)


class RedactorUnit(unittest.TestCase):
    def test_basic_patterns(self):
        cases = {
            "mail jane.doe@example.com now": f"mail {REDACTION} now",
            "ssn 123-45-6789.": f"ssn {REDACTION}.",
            "ssn SSN-123-45-6789 ok": f"ssn SSN-{REDACTION} ok",
            "card 4111 1111 1111 1111!": f"card {REDACTION}!",
            "card 4111-1111-1111-1111": f"card {REDACTION}",
            "card 4111111111111111": f"card {REDACTION}",
            "amex 3782 822463 10005 ok": f"amex {REDACTION} ok",
        }
        for src, want in cases.items():
            with self.subTest(src=src):
                self.assertEqual(redact_text(src), want)
                self.assertEqual(stream_all(src, lambda: 1), want)

    def test_non_pii_untouched(self):
        for s in ["order 20240117 total 1,234.56", "call 555-1234", "1234 5678 9012 3456", "v2.0.1 on 2024-01-17",
                  "user@localhost", "9" * 25, "a@b", "Hello, world. 1 2 3 4"]:
            with self.subTest(s=s):
                self.assertEqual(stream_all(s, lambda: 1), s)

    def test_card_next_to_other_numbers_is_still_found(self):
        # regression: a greedy "candidate" that failed Luhn used to swallow the real card
        for s in ["4111111111111111 1,234.56", "order 12345 4111 1111 1111 1111", "call 555-1234 4111 1111 1111 1111"]:
            with self.subTest(s=s):
                self.assertIn(REDACTION, redact_text(s))
                self.assertNotIn("4111", stream_all(s, lambda: 2))

    def test_every_possible_split_point(self):
        text = "x jane.doe@example.com y 123-45-6789 z 4111 1111 1111 1111 w a+b@mail.co.uk."
        want = redact_text(text)
        for a in range(len(text) + 1):
            for b in range(a, len(text) + 1):
                r = StreamRedactor()
                got = r.feed(text[:a]) + r.feed(text[a:b]) + r.feed(text[b:]) + r.flush()
                self.assertEqual(got, want, (a, b))

    def test_random_chunking_equals_whole_text(self):
        rnd = random.Random(7)
        pii = ["jane.doe@example.com", "123-45-6789", "4111 1111 1111 1111", "4111111111111111", "SSN-123-45-6789"]
        benign = ["Hello", "order 20240117", "1,234.56", "12 34", "foo@bar", ".com", "-", "2024-01-17", "12345", "total 19.99"]
        seps = [" ", "\n", ", ", ". ", ": ", " - ", "(", ") "]
        for _ in range(4000):
            parts = [rnd.choice(pii + benign) for _ in range(rnd.randint(1, 8))]
            text = parts[0] + "".join(rnd.choice(seps) + p for p in parts[1:])
            got = stream_all(text, lambda: rnd.choice([1, 1, 2, 3, 5, 8, 40]))
            self.assertEqual(got, redact_text(text), text)
            for p in pii:
                self.assertNotIn(p.replace("SSN-", ""), got)

    def test_holdback_is_bounded_and_progress_is_made(self):
        rnd = random.Random(1)
        chunk = "The quick brown fox jumps over the lazy dog. Contact bob@corp.io or 555-12-3456. " * 3
        r = StreamRedactor(); emitted = 0; fed = 0; peak = 0
        for _ in range(4000):
            fed += len(chunk); emitted += len(r.feed(chunk)); peak = max(peak, r.buffered)
        emitted += len(r.flush())
        self.assertLessEqual(peak, 2 * MAX_EMAIL_LEN)
        self.assertGreater(emitted, 0)
        # one pathological long token must not grow memory either
        r = StreamRedactor()
        for _ in range(20000):
            r.feed("A" * 50)
            self.assertLessEqual(r.buffered, 2 * MAX_EMAIL_LEN)

    def test_time_to_first_emit_is_one_word_not_a_fixed_window(self):
        r = StreamRedactor()
        self.assertEqual(r.feed("Hello"), "")        # might still become an email local-part
        self.assertEqual(r.feed(", wor"), "Hello, ") # released as soon as it can't be
        self.assertEqual(r.flush(), "wor")

    def test_throughput(self):
        text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit. Mail ann@example.com about 1,234.56 USD. " * 10000)
        t = time.perf_counter(); stream_all(text, lambda: 6); dt = time.perf_counter() - t
        print(f"\n  throughput: {len(text)/1e6:.1f} MB in {dt:.2f}s = {len(text)/1e6/dt:.1f} MB/s (6-char deltas)")
        self.assertLess(dt, 30)


class GatewayIntegration(unittest.IsolatedAsyncioTestCase):
    async def _serve(self, provider_factory):
        srv = await llm_gateway.LLMGateway(provider_factory).start()
        self.addCleanup(srv.close)
        return srv.sockets[0].getsockname()[1]

    @staticmethod
    async def _request(port, body=b'{"prompt":"hi"}', path="/v1/generate", read_chunks=None):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        t0 = time.perf_counter()
        writer.write(f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Length: {len(body)}\r\n\r\n".encode() + body)
        await writer.drain()
        head = (await reader.readuntil(b"\r\n\r\n")).decode()
        frames, ttft = [], None
        if "chunked" in head:
            while True:
                size = int((await reader.readline()).strip() or b"0", 16)
                if size == 0:
                    break
                data = await reader.readexactly(size); await reader.readexactly(2)
                ttft = ttft if ttft is not None else time.perf_counter() - t0
                frames.append(data.decode())
                if read_chunks and len(frames) >= read_chunks:
                    writer.close()  # abrupt client disconnect
                    return head, frames, ttft, time.perf_counter() - t0
        else:
            frames.append((await reader.read()).decode())
        writer.close()
        return head, frames, ttft, time.perf_counter() - t0

    @staticmethod
    def _text(frames):
        out = ""
        for f in frames:
            if f.startswith("data: ") and "[DONE]" not in f:
                out += json.loads(f[6:].strip())["delta"]
        return out

    async def test_end_to_end_redaction(self):
        port = await self._serve(lambda: llm_gateway.MockProvider(chunk=3))
        head, frames, _, _ = await self._request(port)
        text = self._text(frames)
        self.assertIn("200 OK", head); self.assertIn("text/event-stream", head)
        self.assertEqual(text, redact_text(llm_gateway.MockProvider.TEXT))
        for raw in ["jane.doe@example.com", "john_smith+test@mail.co.uk", "123-45-6789", "4111 1111 1111 1111"]:
            self.assertNotIn(raw, "".join(frames))
        self.assertIn("order 20240117", text)
        self.assertTrue(frames[-1].startswith("data: [DONE]"))

    async def test_first_token_arrives_long_before_stream_ends(self):
        port = await self._serve(lambda: llm_gateway.MockProvider(chunk=4, delay_s=0.02))
        _, frames, ttft, total = await self._request(port)
        print(f"\n  TTFT {ttft*1000:.0f} ms, total {total*1000:.0f} ms, {len(frames)} frames")
        self.assertLess(ttft, 0.25)
        self.assertGreater(total, 0.8)  # i.e. we really are streaming, not buffering to the end

    async def test_client_disconnect_stops_the_provider(self):
        closed = asyncio.Event()

        class Endless:
            async def stream(self, prompt):
                try:
                    while True:
                        await asyncio.sleep(0.005); yield "token "
                finally:
                    closed.set()

        port = await self._serve(Endless)
        await self._request(port, read_chunks=3)
        await asyncio.wait_for(closed.wait(), 2)

    async def test_mid_stream_provider_failure_is_sanitised(self):
        class Boom:
            async def stream(self, prompt):
                yield "partial answer "
                yield "more "
                raise RuntimeError("db password=hunter2 host=10.0.0.5")

        port = await self._serve(Boom)
        _, frames, _, _ = await self._request(port)
        blob = "".join(frames)
        self.assertIn("event: error", blob); self.assertIn("upstream_error", blob)
        self.assertNotIn("hunter2", blob); self.assertNotIn("10.0.0.5", blob)

    async def test_bad_requests(self):
        port = await self._serve(llm_gateway.MockProvider)
        for body, path, status in [(b"nope", "/v1/generate", "400"), (b'{"prompt": 5}', "/v1/generate", "400"),
                                   (b'{"prompt":"x"}', "/other", "404")]:
            head, _, _, _ = await self._request(port, body=body, path=path)
            self.assertIn(status, head.splitlines()[0])

    async def test_concurrent_streams_are_independent(self):
        port = await self._serve(lambda: llm_gateway.MockProvider(chunk=2))
        results = await asyncio.gather(*[self._request(port) for _ in range(40)])
        want = redact_text(llm_gateway.MockProvider.TEXT)
        self.assertTrue(all(self._text(r[1]) == want for r in results))

    async def test_redact_stream_wrapper(self):
        async def src():
            for part in ["mail me at ann", "@exam", "ple.com ok"]:
                yield part
        out = [x async for x in redact_stream(src())]
        self.assertEqual("".join(out), f"mail me at {REDACTION} ok")


if __name__ == "__main__":
    unittest.main(verbosity=2)
