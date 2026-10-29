import asyncio
import json
import logging
import os
import tempfile
import time
import unittest

from router import (CompletionRequest, GatewayRouter, ProviderResult, SlidingWindowTokenLimiter, UpstreamClientError,
                    UpstreamRateLimited, UpstreamServerError, Allowed, Denied, TooLarge, estimate_tokens)

logging.disable(logging.CRITICAL)  # keep test output clean; errors are asserted via responses


class FakeClock:
    def __init__(self, t=1_000_000.0): self.t = t
    def __call__(self): return self.t
    def advance(self, s): self.t += s


def ok(model="m", p=10, c=20):
    return ProviderResult(content="hello", prompt_tokens=p, completion_tokens=c, model=model)


class Scripted:
    """Provider whose behaviour is a coroutine factory."""
    def __init__(self, behaviour): self.behaviour, self.calls = behaviour, 0
    async def complete(self, req):
        self.calls += 1
        return await self.behaviour()


def returns(result=None):
    async def f(): return result or ok()
    return f

def raises(exc):
    async def f(): raise exc
    return f

def sleeps(seconds, then=None):
    async def f():
        await asyncio.sleep(seconds); return then or ok()
    return f

def ignores_cancellation(seconds):
    async def f():
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            await asyncio.sleep(seconds)   # misbehaving provider: swallows cancel and keeps going
        return ok()
    return f


class DbCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.path = os.path.join(self.tmp, "limits.db")
        self.clock = FakeClock()
        self.lim = SlidingWindowTokenLimiter(self.path, 50_000, 60, clock=self.clock)
        self.addCleanup(self.lim.close)


class LimiterTests(DbCase):
    async def test_uses_an_on_disk_sqlite_file(self):
        await self.lim.reserve("t", 100)
        self.assertTrue(os.path.getsize(self.path) > 0)

    async def test_window_slides_and_evicts(self):
        self.assertIsInstance(await self.lim.reserve("t", 30_000), Allowed)       # t=0
        self.clock.advance(30)
        self.assertIsInstance(await self.lim.reserve("t", 20_000), Allowed)       # t=30 -> 50k used
        self.clock.advance(1)
        d = await self.lim.reserve("t", 1_000)                                    # t=31
        self.assertIsInstance(d, Denied)
        self.assertEqual(d.retry_after_ms, 29_000)                                # the 30k row leaves at t=60
        self.clock.advance(28.999)
        self.assertIsInstance(await self.lim.reserve("t", 1_000), Denied)         # t=59.999 still inside
        self.clock.advance(0.001)
        self.assertIsInstance(await self.lim.reserve("t", 1_000), Allowed)        # t=60.000 evicted
        self.assertEqual(await self.lim.used("t"), 21_000)
        self.assertEqual(await self.lim.row_count(), 2)                           # old row physically deleted

    async def test_retry_after_waits_for_enough_capacity_not_just_the_oldest_row(self):
        await self.lim.reserve("t", 10_000); self.clock.advance(10)
        await self.lim.reserve("t", 10_000); self.clock.advance(10)
        await self.lim.reserve("t", 30_000)                                       # t=20, 50k used
        d = await self.lim.reserve("t", 25_000)                                   # needs 25k freed; first two rows only free 20k
        # naive "wait for the oldest row" would say 40 s (t=60); the correct answer waits for the
        # third row, written at t=20, which leaves the window at t=80 -> 60 s from now (t=20)
        self.assertEqual(d.retry_after_ms, 60_000)

    async def test_exact_limit_allowed_one_over_denied(self):
        self.assertIsInstance(await self.lim.reserve("t", 50_000), Allowed)
        self.assertIsInstance(await self.lim.reserve("t", 1), Denied)

    async def test_tenants_are_isolated(self):
        await self.lim.reserve("a", 50_000)
        self.assertIsInstance(await self.lim.reserve("b", 50_000), Allowed)

    async def test_request_larger_than_limit_is_toolarge_not_retryable(self):
        self.assertIsInstance(await self.lim.reserve("t", 50_001), TooLarge)

    async def test_reconcile_and_release(self):
        a = await self.lim.reserve("t", 50_000)
        await self.lim.adjust(a.reservation_id, 10_000)                           # provider used far less
        b = await self.lim.reserve("t", 40_000)
        self.assertIsInstance(b, Allowed)
        await self.lim.release(b.reservation_id)
        self.assertEqual(await self.lim.used("t"), 10_000)

    async def test_concurrent_reservations_are_atomic(self):
        results = await asyncio.gather(*[self.lim.reserve("t", 1_000) for _ in range(120)])
        self.assertEqual(sum(isinstance(r, Allowed) for r in results), 50)         # exactly limit/1000, never 51
        self.assertEqual(await self.lim.used("t"), 50_000)

    async def test_two_processes_sharing_one_db_file_cannot_overspend(self):
        other = SlidingWindowTokenLimiter(self.path, 50_000, 60, clock=self.clock)
        self.addCleanup(other.close)
        results = await asyncio.gather(*[(self.lim if i % 2 else other).reserve("t", 1_000) for i in range(120)])
        self.assertEqual(sum(isinstance(r, Allowed) for r in results), 50)

    async def test_state_survives_restart(self):
        await self.lim.reserve("t", 40_000)
        self.lim.close()
        again = SlidingWindowTokenLimiter(self.path, 50_000, 60, clock=self.clock)
        self.addCleanup(again.close)
        self.assertIsInstance(await again.reserve("t", 20_000), Denied)

    async def test_global_purge_removes_idle_tenants(self):
        for i in range(20): await self.lim.reserve(f"idle{i}", 10)
        self.clock.advance(61)
        await self.lim.reserve("someone", 10)                                      # triggers purge
        self.assertEqual(await self.lim.row_count(), 1)


class RouterTests(DbCase):
    def make(self, primary, secondary, **kw):
        return GatewayRouter(primary, secondary, self.lim, **kw)

    def req(self, **kw):
        return CompletionRequest(**{"api_key": "sk-tenant-1", "prompt": "hello there", "max_tokens": 100, **kw})

    async def test_happy_path_uses_primary_and_reconciles_tokens(self):
        p, s = Scripted(returns()), Scripted(returns())
        r = await self.make(p, s).handle(self.req())
        self.assertEqual(r.status, 200); self.assertFalse(r.body["fallback_used"])
        self.assertEqual((p.calls, s.calls), (1, 0))
        self.assertEqual(await self.lim.used(self.lim.tenant_id("sk-tenant-1")), 30)  # actual usage, not the estimate

    async def test_429_fails_over(self):
        p, s = Scripted(raises(UpstreamRateLimited())), Scripted(returns(ok("backup")))
        r = await self.make(p, s).handle(self.req())
        self.assertEqual((r.status, r.body["model"], r.body["fallback_used"]), (200, "backup", True))
        self.assertEqual(r.headers["X-Gateway-Fallback"], "1")

    async def test_timeout_fails_over_at_3000ms(self):
        p, s = Scripted(sleeps(10)), Scripted(returns(ok("backup")))
        t = time.perf_counter()
        r = await self.make(p, s).handle(self.req())      # default 3.0 s
        dt = time.perf_counter() - t
        self.assertEqual(r.body["model"], "backup")
        self.assertTrue(3.0 <= dt < 3.5, dt)

    async def test_provider_that_ignores_cancellation_cannot_delay_failover(self):
        p, s = Scripted(ignores_cancellation(2)), Scripted(returns(ok("backup")))
        t = time.perf_counter()
        r = await self.make(p, s, primary_timeout_s=0.2).handle(self.req())
        self.assertEqual(r.body["model"], "backup")
        self.assertLess(time.perf_counter() - t, 1.0)     # wait_for() would have blocked ~2s here

    async def test_late_primary_result_is_ignored(self):
        p, s = Scripted(sleeps(0.3, ok("late-primary"))), Scripted(returns(ok("backup")))
        r = await self.make(p, s, primary_timeout_s=0.1).handle(self.req())
        await asyncio.sleep(0.4)
        self.assertEqual(r.body["model"], "backup")

    async def test_5xx_and_unexpected_exceptions_fail_over(self):
        for exc in (UpstreamServerError(), RuntimeError("boom")):
            r = await self.make(Scripted(raises(exc)), Scripted(returns(ok("b")))).handle(self.req())
            self.assertEqual(r.body["model"], "b")

    async def test_4xx_from_upstream_does_not_fail_over(self):
        s = Scripted(returns())
        r = await self.make(Scripted(raises(UpstreamClientError())), s).handle(self.req())
        self.assertEqual((r.status, r.body["error"]["code"], s.calls), (502, "upstream_rejected", 0))

    async def test_both_down_gives_standard_503_and_refunds_tokens(self):
        r = await self.make(Scripted(raises(UpstreamRateLimited())), Scripted(raises(UpstreamServerError()))).handle(self.req())
        self.assertEqual(r.status, 503)
        self.assertEqual(set(r.body["error"]), {"type", "code", "message", "request_id"})
        self.assertEqual(r.body["error"]["request_id"], r.headers["X-Request-Id"])
        self.assertEqual(await self.lim.used(self.lim.tenant_id("sk-tenant-1")), 0)

    async def test_both_timing_out_gives_504(self):
        r = await self.make(Scripted(sleeps(5)), Scripted(sleeps(5)), primary_timeout_s=0.05, secondary_timeout_s=0.05).handle(self.req())
        self.assertEqual((r.status, r.body["error"]["code"]), (504, "upstream_timeout"))

    async def test_errors_never_leak_internals(self):
        secret = "Traceback File /srv/app/provider.py line 9 password=hunter2 host=db-prod-7:5432 sk-live-abc"
        for exc in (RuntimeError(secret), UpstreamServerError(secret), UpstreamRateLimited(secret)):
            r = await self.make(Scripted(raises(exc)), Scripted(raises(RuntimeError(secret)))).handle(self.req())
            blob = json.dumps(r.body) + json.dumps(r.headers)
            for needle in ("Traceback", "hunter2", "db-prod", "/srv/app", "sk-live", "RuntimeError", "File "):
                self.assertNotIn(needle, blob)
            self.assertEqual(r.body["error"]["type"], "gateway_error")

    async def test_rate_limit_response_has_retry_after_and_skips_providers(self):
        p, s = Scripted(returns()), Scripted(returns())
        router = self.make(p, s)
        await self.lim.reserve(self.lim.tenant_id("sk-tenant-1"), 49_990)
        r = await router.handle(self.req())
        self.assertEqual((r.status, r.body["error"]["code"]), (429, "rate_limit_exceeded"))
        self.assertEqual(r.body["error"]["retry_after_ms"], 60_000)
        self.assertEqual(r.headers["Retry-After"], "60")
        self.assertEqual((p.calls, s.calls), (0, 0))

    async def test_oversized_request_is_413(self):
        r = await self.make(Scripted(returns()), Scripted(returns())).handle(self.req(max_tokens=32_000, prompt="x" * 100_000))
        self.assertEqual(r.status, 413)

    async def test_input_validation(self):
        router = self.make(Scripted(returns()), Scripted(returns()))
        for bad in (dict(prompt=""), dict(max_tokens=0), dict(max_tokens=True), dict(api_key=""), dict(max_tokens=10**9), dict(prompt=None)):
            r = await router.handle(self.req(**bad))
            self.assertEqual((r.status, r.body["error"]["code"]), (400, "invalid_request"), bad)

    async def test_client_cancellation_cancels_upstream_and_refunds(self):
        started = asyncio.Event(); cancelled = asyncio.Event()
        async def slow():
            started.set()
            try: await asyncio.sleep(30)
            except asyncio.CancelledError: cancelled.set(); raise
        router = self.make(Scripted(slow), Scripted(returns()))
        task = asyncio.create_task(router.handle(self.req()))
        await started.wait(); task.cancel()
        with self.assertRaises(asyncio.CancelledError): await task
        await asyncio.wait_for(cancelled.wait(), 1)
        self.assertEqual(await self.lim.used(self.lim.tenant_id("sk-tenant-1")), 0)

    async def test_limiter_store_failure_fails_closed_without_leaking(self):
        router = self.make(Scripted(returns()), Scripted(returns()))
        self.lim._db.close()                                   # simulate a broken database
        r = await router.handle(self.req())
        self.assertEqual((r.status, r.body["error"]["code"]), (503, "service_unavailable"))
        self.assertNotIn("sqlite", json.dumps(r.body).lower())
        self.lim._db = __import__("sqlite3").connect(":memory:", check_same_thread=False)  # let cleanup close cleanly

    async def test_many_concurrent_requests_with_flaky_primary(self):
        n = {"i": 0}
        async def flaky():
            n["i"] += 1
            if n["i"] % 3 == 0: raise UpstreamRateLimited()
            await asyncio.sleep(0.01); return ok()
        router = self.make(Scripted(flaky), Scripted(returns(ok("backup"))))
        rs = await asyncio.gather(*[router.handle(self.req(api_key=f"k{i%5}")) for i in range(150)])
        self.assertTrue(all(r.status == 200 for r in rs))
        self.assertGreater(sum(r.body["fallback_used"] for r in rs), 30)

    def test_estimate(self):
        self.assertEqual(estimate_tokens(CompletionRequest("k", "a" * 400, 100)), 200)


if __name__ == "__main__":
    unittest.main(verbosity=2)
