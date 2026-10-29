"""
Incremental PII redactor for token streams.

The problem
-----------
An LLM streams text in small deltas, and PII can be split across any boundary:

    "...mail jane.do" | "e@exam" | "ple.com please"

Redacting each delta independently leaks it. Buffering the whole response fixes
that but destroys time-to-first-token and uses O(response) memory.

The approach: bounded, *context-aware* holdback
-----------------------------------------------
After appending a delta, we emit everything that can no longer become part of a
match and keep only the tail that still might. Each pattern's characters form a
simple character class, so "could still grow into a match" == "the buffer ends
inside a run of those characters":

  * email : trailing run of  [A-Za-z0-9._%+-@]   (capped at MAX_EMAIL_LEN)
  * SSN / card : trailing run of  [0-9 -]        (capped at MAX_CARD_LEN)

The cut point is the start of the earliest such run. Consequences:
  * latency cost is at most one "word" (the run currently being typed), not a
    fixed N-character window, so TTFT is only delayed by roughly one delta
  * memory is O(max match length) (< 700 chars), independent of response size
  * a final `flush()` redacts and releases whatever is left at end-of-stream

Safety net: the cut is never allowed to fall inside a complete match; if it
would, it is moved back to that match's start. (Only reachable for runs longer
than the caps, i.e. pathological input.)

Credit cards: every window of 13-19 digits made of whole digit groups (joined by
single spaces/dashes) is Luhn-checked, so order numbers/timestamps are left alone
while a card sitting next to another number ("4111111111111111 1,234.56") is
still found. All pattern hits are unioned and merged, so overlaps can only
cause *more* redaction, never a leak.

Where chunk boundaries are concerned, results are identical to redacting the
complete text in one go (verified by a randomised property test).
"""
from __future__ import annotations

import re
from typing import AsyncIterable, AsyncIterator, Iterator

REDACTION = "[REDACTED]"

_EMAIL = re.compile(r"[A-Za-z0-9._%+-]{1,64}@(?:[A-Za-z0-9-]{1,63}\.){1,4}[A-Za-z]{2,24}")
# Dashed SSN. Only digits (not '-') guard the edges so "SSN-123-45-6789" is still caught.
_SSN = re.compile(r"(?<![0-9])[0-9]{3}-[0-9]{2}-[0-9]{4}(?![0-9])")
# A "chain" is digit groups joined by single spaces/dashes; card numbers are Luhn-valid
# windows of whole groups inside a chain (see _card_spans).
_CHAIN = re.compile(r"(?<![0-9])[0-9]+(?:[ -][0-9]+)*(?![0-9])")
_GROUP = re.compile(r"[0-9]+")

_EMAIL_CHARS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789._%+-@")
_DIGIT_RUN_CHARS = frozenset("0123456789 -")
_DIGITS = frozenset("0123456789")

MAX_EMAIL_LEN = 64 + 1 + 4 * 64 + 24  # longest string _EMAIL can match (345)
MAX_CARD_LEN = 19 + 18  # 19 digits + 18 single separators (37)


def _luhn_ok(candidate: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed([c for c in candidate if c in _DIGITS])):
        d = ord(ch) - 48
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _card_spans(text: str) -> Iterator[tuple[int, int]]:
    """
    Every Luhn-valid window of 13-19 digits made of whole digit groups.

    Scanning *all* windows (instead of one greedy match) is what stops a
    neighbouring number from hiding a real card, e.g. "4111111111111111 1,234.56"
    or "order 12345 4111 1111 1111 1111".
    """
    for chain in _CHAIN.finditer(text):
        if chain.end() - chain.start() < 13:
            continue
        base = chain.start()
        groups = [(g.start() + base, g.end() + base) for g in _GROUP.finditer(chain.group())]
        for i, (gs, _) in enumerate(groups):
            digits = 0
            for j in range(i, len(groups)):
                digits += groups[j][1] - groups[j][0]
                if digits > 19:
                    break
                if digits >= 13 and _luhn_ok(text[gs : groups[j][1]]):
                    yield gs, groups[j][1]


def _spans(text: str) -> list[tuple[int, int]]:
    """Ordered, merged (start, end) spans of PII in `text`. Overlaps are unioned, never dropped."""
    spans = [m.span() for m in _EMAIL.finditer(text)]
    spans.extend(m.span() for m in _SSN.finditer(text))
    spans.extend(_card_spans(text))
    if not spans:
        return []
    spans.sort()
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            if e > merged[-1][1]:
                merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(s, e) for s, e in merged]


def redact_text(text: str) -> str:
    """Non-streaming reference implementation."""
    out: list[str] = []
    last = 0
    for s, e in _spans(text):
        out.append(text[last:s])
        out.append(REDACTION)
        last = e
    out.append(text[last:])
    return "".join(out)


def _trailing_run_start(buf: str, charset: frozenset[str], cap: int) -> int:
    """Index where the run of `charset` chars ending at len(buf) begins (<= cap chars back)."""
    n = len(buf)
    i = n
    limit = max(0, n - cap)
    while i > limit and buf[i - 1] in charset:
        i -= 1
    return i


class StreamRedactor:
    """Feed deltas with `feed()`, call `flush()` once at end of stream."""

    __slots__ = ("_buf",)

    def __init__(self) -> None:
        self._buf = ""

    @property
    def buffered(self) -> int:
        return len(self._buf)

    def feed(self, delta: str) -> str:
        """Consume a delta; return the text that is now safe to send (possibly "")."""
        if not delta:
            return ""
        buf = self._buf + delta if self._buf else delta
        cut = self._safe_cut(buf)
        if cut <= 0:
            self._buf = buf
            return ""
        self._buf = buf[cut:]
        return redact_text(buf[:cut])

    def flush(self) -> str:
        """End of stream: redact and release whatever is still held back."""
        out, self._buf = redact_text(self._buf), ""
        return out

    @staticmethod
    def _safe_cut(buf: str) -> int:
        n = len(buf)
        cut = n

        # Could the tail still become an email? (any word could be a local-part)
        e = _trailing_run_start(buf, _EMAIL_CHARS, MAX_EMAIL_LEN)
        if e < n:
            cut = e

        # Could the tail still become an SSN / card number?
        d = _trailing_run_start(buf, _DIGIT_RUN_CHARS, MAX_CARD_LEN)
        if 0 < d < n and buf[d - 1] in _DIGITS and buf[d] in _DIGITS:
            # the cap landed mid-group; a window is made of whole groups and any
            # group straddling the cap starts too early to reach the end, so skip it
            while d < n and buf[d] in _DIGITS:
                d += 1
        while d < n and buf[d] not in _DIGITS:  # a match must start with a digit
            d += 1
        if d < n and d < cut:
            cut = d

        # Never cut through a complete match.
        if 0 < cut < n:
            for s, e2 in _spans(buf):
                if s >= cut:
                    break
                if e2 > cut:
                    cut = s
                    break
        return cut


async def redact_stream(source: AsyncIterable[str], redactor: StreamRedactor | None = None) -> AsyncIterator[str]:
    """Wrap an async iterator of text deltas; yields redacted deltas as soon as they are safe."""
    r = redactor or StreamRedactor()
    async for delta in source:
        out = r.feed(delta)
        if out:
            yield out
    tail = r.flush()
    if tail:
        yield tail
