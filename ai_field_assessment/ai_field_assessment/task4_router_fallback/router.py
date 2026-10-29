"""
Resilient routing for an LLM gateway:

    request -> [token-aware sliding-window limiter (SQLite)] -> primary --429/timeout--> secondary
                                                                  \\__ standardised, sanitised errors

Rate limiter
  * Sliding-window *log*: one row per request (tenant, ts_ms, tokens). Usage is the
    exact sum over (now - window, now], so there is no fixed-window burst at the edge.
  * Check-and-reserve is a single `BEGIN IMMEDIATE` transaction => atomic across
    asyncio tasks AND across several gateway processes sharing the same DB file.
  * Tokens are *estimated up front* (prompt/4 + max_tokens), reserved, then
    reconciled to the provider's real usage; refunded if the request fails.
  * Eviction: expired rows for the tenant are deleted inside every reservation, plus
    a global purge at most once per window, so the table stays small.
  * Tenant ids are SHA-256 of the API key: raw keys are never written to disk.
  * SQLite work runs on one dedicated thread (never blocks the event loop; also
    serialises access to the connection). If the limiter store fails we fail
    *closed* (503) to protect upstream spend; flip that policy in `handle()` if you
    prefer availability.

Failover
  * The primary attempt is raced against a 3000 ms deadline using asyncio.wait on a
    task, NOT wait_for: a provider that swallows cancellation can't hold the request
    hostage past the deadline. The abandoned task is cancelled and its eventual
    result/exception is consumed so nothing is logged as "never retrieved".
  * 429, timeouts, 5xx and transport errors fail over. Other 4xx do not (the same
    request would fail again) and surface as a gateway error.
  * One reservation per client request regardless of how many upstream attempts.

Errors
  * Every non-2xx is  {"error": {"type","code","message","request_id"[,"retry_after_ms"]}}
    with fixed, human-safe messages. Upstream bodies, exception text and stack traces
    are logged server-side only.
"""
from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import sqlite3
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

log = logging.getLogger("llm-router")

# --------------------------------------------------------------------------- #
# Provider contract
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CompletionRequest:
    api_key: str
    prompt: str
    max_tokens: int = 256
    model: str | None = None


@dataclass(frozen=True)
class ProviderResult:
    content: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class UpstreamRateLimited(Exception):
    """Provider answered 429."""


class UpstreamServerError(Exception):
    """Provider answered 5xx / transport failure."""


class UpstreamClientError(Exception):
    """Provider rejected the request (4xx other than 429); retrying elsewhere won't help."""


class Provider(Protocol):
    async def complete(self, req: CompletionRequest) -> ProviderResult: ...


# --------------------------------------------------------------------------- #
# Limiter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Allowed:
    reservation_id: int
    remaining: int


@dataclass(frozen=True)
class Denied:
    retry_after_ms: int
    used: int


@dataclass(frozen=True)
class TooLarge:
    limit: int


class SlidingWindowTokenLimiter:
    def __init__(
        self,
        db_path: str,
        limit_tokens: int = 50_000,
        window_s: float = 60.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self.limit = limit_tokens
        self.window_ms = int(window_s * 1000)
        self._clock = clock
        self._last_purge_ms = 0
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="limiter-db")
        self._db = sqlite3.connect(db_path, timeout=5.0, isolation_level=None, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.execute("PRAGMA busy_timeout=5000")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS usage (
                id     INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant TEXT    NOT NULL,
                ts_ms  INTEGER NOT NULL,
                tokens INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS usage_tenant_ts ON usage(tenant, ts_ms);
            CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts_ms);
            """
        )

    @staticmethod
    def tenant_id(api_key: str) -> str:
        return hashlib.sha256(api_key.encode()).hexdigest()[:32]

    def _now_ms(self) -> int:
        return int(self._clock() * 1000)

    # ---- sync internals (run on the DB thread) ---------------------------- #
    def _reserve_sync(self, tenant: str, tokens: int) -> Allowed | Denied | TooLarge:
        if tokens > self.limit:
            return TooLarge(self.limit)  # can never fit, so don't tell the caller to "retry later"
        now = self._now_ms()
        cutoff = now - self.window_ms
        db = self._db
        db.execute("BEGIN IMMEDIATE")  # take the write lock before reading => atomic check-and-insert
        try:
            db.execute("DELETE FROM usage WHERE tenant=? AND ts_ms<=?", (tenant, cutoff))
            if now - self._last_purge_ms >= self.window_ms:
                db.execute("DELETE FROM usage WHERE ts_ms<=?", (cutoff,))
                self._last_purge_ms = now
            used = db.execute("SELECT COALESCE(SUM(tokens),0) FROM usage WHERE tenant=?", (tenant,)).fetchone()[0]
            if used + tokens > self.limit:
                need = used + tokens - self.limit
                retry_ms, freed = self.window_ms, 0
                for ts, tk in db.execute("SELECT ts_ms, tokens FROM usage WHERE tenant=? ORDER BY ts_ms, id", (tenant,)):
                    freed += tk
                    if freed >= need:
                        retry_ms = ts + self.window_ms - now  # moment this row leaves the window
                        break
                db.execute("COMMIT")  # keep the evictions
                return Denied(retry_after_ms=max(1, retry_ms), used=used)
            cur = db.execute("INSERT INTO usage(tenant, ts_ms, tokens) VALUES (?,?,?)", (tenant, now, tokens))
            db.execute("COMMIT")
            return Allowed(reservation_id=cur.lastrowid, remaining=self.limit - used - tokens)
        except BaseException:
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def _adjust_sync(self, rid: int, tokens: int) -> None:
        self._db.execute("UPDATE usage SET tokens=? WHERE id=?", (max(0, tokens), rid))

    def _release_sync(self, rid: int) -> None:
        self._db.execute("DELETE FROM usage WHERE id=?", (rid,))

    def _used_sync(self, tenant: str) -> int:
        cutoff = self._now_ms() - self.window_ms
        return self._db.execute(
            "SELECT COALESCE(SUM(tokens),0) FROM usage WHERE tenant=? AND ts_ms>?", (tenant, cutoff)
        ).fetchone()[0]

    def _row_count_sync(self) -> int:
        return self._db.execute("SELECT COUNT(*) FROM usage").fetchone()[0]

    # ---- async API -------------------------------------------------------- #
    async def _run(self, fn: Callable[..., Any], *args: Any) -> Any:
        return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)

    async def reserve(self, tenant: str, tokens: int) -> Allowed | Denied | TooLarge:
        return await self._run(self._reserve_sync, tenant, tokens)

    async def adjust(self, reservation_id: int, actual_tokens: int) -> None:
        await self._run(self._adjust_sync, reservation_id, actual_tokens)

    async def release(self, reservation_id: int) -> None:
        await self._run(self._release_sync, reservation_id)

    async def used(self, tenant: str) -> int:
        return await self._run(self._used_sync, tenant)

    async def row_count(self) -> int:
        return await self._run(self._row_count_sync)

    def close(self) -> None:
        self._pool.shutdown(wait=True)
        self._db.close()


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #
@dataclass
class GatewayResult:
    status: int
    body: dict[str, Any]
    headers: dict[str, str] = field(default_factory=dict)


class _AttemptFailed(Exception):
    def __init__(self, reason: str, retryable: bool) -> None:
        self.reason, self.retryable = reason, retryable


class _GatewayError(Exception):
    def __init__(self, status: int, code: str, message: str, retry_after_ms: int | None = None) -> None:
        self.status, self.code, self.message, self.retry_after_ms = status, code, message, retry_after_ms


def estimate_tokens(req: CompletionRequest) -> int:
    return max(1, math.ceil(len(req.prompt.encode()) / 4)) + req.max_tokens


def _consume(task: "asyncio.Future[Any]") -> None:
    if not task.cancelled():
        task.exception()  # mark as retrieved; late failures of abandoned attempts are expected


class GatewayRouter:
    def __init__(
        self,
        primary: Provider,
        secondary: Provider,
        limiter: SlidingWindowTokenLimiter,
        primary_timeout_s: float = 3.0,
        secondary_timeout_s: float = 30.0,
        max_prompt_chars: int = 200_000,
        max_max_tokens: int = 32_768,
    ) -> None:
        self.primary, self.secondary, self.limiter = primary, secondary, limiter
        self.primary_timeout_s, self.secondary_timeout_s = primary_timeout_s, secondary_timeout_s
        self.max_prompt_chars, self.max_max_tokens = max_prompt_chars, max_max_tokens

    # ---- public ------------------------------------------------------------ #
    async def handle(self, req: CompletionRequest) -> GatewayResult:
        request_id = uuid.uuid4().hex
        reservation: int | None = None
        try:
            self._validate(req)
            tenant = self.limiter.tenant_id(req.api_key)
            try:
                decision = await self.limiter.reserve(tenant, estimate_tokens(req))
            except Exception:
                log.exception("[%s] rate limiter store failed", request_id)
                raise _GatewayError(503, "service_unavailable", "The gateway is temporarily unavailable.") from None

            if isinstance(decision, TooLarge):
                raise _GatewayError(413, "request_too_large", "Request exceeds the per-minute token limit.")
            if isinstance(decision, Denied):
                raise _GatewayError(429, "rate_limit_exceeded", "Token rate limit exceeded.", decision.retry_after_ms)
            reservation = decision.reservation_id

            result, fallback_used = await self._call_with_failover(req, request_id)

            actual = result.prompt_tokens + result.completion_tokens
            try:
                await self.limiter.adjust(reservation, actual)  # reconcile estimate -> reality
            except Exception:
                log.exception("[%s] failed to reconcile token usage", request_id)
            reservation = None  # now committed
            return GatewayResult(
                200,
                {
                    "request_id": request_id,
                    "content": result.content,
                    "model": result.model,
                    "usage": {"prompt_tokens": result.prompt_tokens, "completion_tokens": result.completion_tokens},
                    "fallback_used": fallback_used,
                },
                {"X-Request-Id": request_id, **({"X-Gateway-Fallback": "1"} if fallback_used else {})},
            )
        except _GatewayError as e:
            return self._error(request_id, e)
        except asyncio.CancelledError:
            raise  # `finally` still refunds
        except Exception:
            log.exception("[%s] unexpected gateway failure", request_id)
            return self._error(request_id, _GatewayError(500, "internal_error", "Internal gateway error."))
        finally:
            if reservation is not None:  # failed or cancelled: give the tokens back
                try:
                    await asyncio.shield(self.limiter.release(reservation))
                except Exception:
                    log.exception("[%s] failed to release reservation", request_id)

    # ---- internals --------------------------------------------------------- #
    def _validate(self, req: CompletionRequest) -> None:
        ok = (
            isinstance(req.api_key, str) and req.api_key
            and isinstance(req.prompt, str) and 0 < len(req.prompt) <= self.max_prompt_chars
            and isinstance(req.max_tokens, int) and not isinstance(req.max_tokens, bool)
            and 0 < req.max_tokens <= self.max_max_tokens
        )
        if not ok:
            raise _GatewayError(400, "invalid_request", "Invalid request parameters.")

    async def _call_with_failover(self, req: CompletionRequest, rid: str) -> tuple[ProviderResult, bool]:
        try:
            return await self._attempt(self.primary, req, self.primary_timeout_s, "primary", rid), False
        except _AttemptFailed as f:
            if not f.retryable:
                raise _GatewayError(502, "upstream_rejected", "The model provider rejected the request.") from None
            log.warning("[%s] primary failed (%s); failing over", rid, f.reason)

        try:
            return await self._attempt(self.secondary, req, self.secondary_timeout_s, "secondary", rid), True
        except _AttemptFailed as f:
            log.error("[%s] secondary failed (%s)", rid, f.reason)
            if f.reason == "timeout":
                raise _GatewayError(504, "upstream_timeout", "The model providers did not respond in time.") from None
            raise _GatewayError(503, "upstream_unavailable", "No model provider is currently available.") from None

    async def _attempt(
        self, provider: Provider, req: CompletionRequest, timeout_s: float, label: str, rid: str
    ) -> ProviderResult:
        task = asyncio.ensure_future(provider.complete(req))
        try:
            done, _ = await asyncio.wait({task}, timeout=timeout_s)
        except asyncio.CancelledError:  # our caller went away
            task.cancel(); task.add_done_callback(_consume)
            raise
        if not done:
            task.cancel(); task.add_done_callback(_consume)  # don't await: it may ignore cancellation
            raise _AttemptFailed("timeout", retryable=True)
        try:
            return task.result()
        except UpstreamRateLimited:
            raise _AttemptFailed("rate_limited", True) from None
        except UpstreamServerError:
            raise _AttemptFailed("upstream_5xx", True) from None
        except UpstreamClientError:
            raise _AttemptFailed("rejected", False) from None
        except Exception:
            log.exception("[%s] %s provider raised", rid, label)  # full detail stays in logs
            raise _AttemptFailed("error", True) from None

    @staticmethod
    def _error(rid: str, e: _GatewayError) -> GatewayResult:
        err: dict[str, Any] = {"type": "gateway_error", "code": e.code, "message": e.message, "request_id": rid}
        headers = {"X-Request-Id": rid}
        if e.retry_after_ms is not None:
            err["retry_after_ms"] = e.retry_after_ms
            headers["Retry-After"] = str(max(1, math.ceil(e.retry_after_ms / 1000)))
        return GatewayResult(e.status, {"error": err}, headers)
