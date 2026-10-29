#!/usr/bin/env python3
"""
MCP security gateway: HTTP/JSON-RPC reverse proxy placed between an agent and a
downstream MCP server (Streamable-HTTP style: one JSON-RPC message per POST).

  agent --Bearer JWT--> [ gateway ] --JSON-RPC--> downstream MCP server

Behaviour
  * Authentication : `Authorization: Bearer <HS256 JWT>`; role comes from the
                     `role` claim ("admin" | "viewer"; anything else => viewer).
                     Missing/invalid/expired token => HTTP 401 (never forwarded).
  * tools/list     : forwarded transparently (optionally hides admin_* tools
                     from non-admins with --filter-tools-list).
  * tools/call     : if params.name starts with "admin_" the caller must be
                     admin, otherwise the gateway answers JSON-RPC error
                     -32001 "Unauthorized Tool Call" itself and the downstream
                     server is never contacted.
  * everything else: forwarded untouched (initialize, ping, notifications...).

Decisions worth calling out
  * The *original bytes* are forwarded (no re-serialisation), so the gateway
    can't accidentally alter a message it approved.
  * Deny rule is normalised (strip + casefold) so "ADMIN_reset_key" or
    " admin_x" can't slip past a case-insensitive downstream.
  * JSON-RPC batches are rejected (-32600): a batch is the classic way to hide
    a forbidden call behind an allowed one, and the current MCP spec no longer
    allows them.
  * A denied *notification* (no id) gets HTTP 202 and no body, because JSON-RPC
    forbids replying to notifications - but it is still not forwarded.
  * HTTP status: transport/envelope problems (-32700, -32600) => 400;
    authorisation failure (-32001) => HTTP 200 carrying the JSON-RPC error,
    so MCP clients surface it as a normal protocol error.
  * The user's Authorization header is NOT forwarded downstream.
"""
from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import hmac
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

log = logging.getLogger("mcp-gateway")

# JSON-RPC / gateway error codes
PARSE_ERROR = -32700
INVALID_REQUEST = -32600
INVALID_PARAMS = -32602
UNAUTHORIZED_TOOL_CALL = -32001
DOWNSTREAM_UNAVAILABLE = -32003

ADMIN_PREFIX = "admin_"
MAX_BODY = 1 << 20  # 1 MiB

HOP_BY_HOP = {
    "connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
    "te", "trailer", "transfer-encoding", "upgrade", "host", "content-length",
    "authorization",  # user's token stays at the gateway
}
RESPONSE_HEADER_ALLOWLIST = {"content-type", "mcp-session-id", "mcp-protocol-version"}


# --------------------------------------------------------------------------- #
# Tokens (HS256 JWT, verified with the stdlib)
# --------------------------------------------------------------------------- #
def _b64url_decode(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def _b64url_encode(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def make_token(secret: bytes, role: str, ttl_s: int = 3600, now: float | None = None) -> str:
    """Helper for tests/demos."""
    now = time.time() if now is None else now
    h = _b64url_encode(json.dumps({"alg": "HS256", "typ": "JWT"}).encode())
    p = _b64url_encode(json.dumps({"role": role, "exp": int(now + ttl_s)}).encode())
    sig = hmac.new(secret, f"{h}.{p}".encode(), hashlib.sha256).digest()
    return f"{h}.{p}.{_b64url_encode(sig)}"


def verify_token(token: str, secret: bytes, now: float | None = None) -> dict[str, Any] | None:
    """Return claims if the token is authentic and unexpired, else None."""
    now = time.time() if now is None else now
    try:
        h_b64, p_b64, s_b64 = token.split(".")
        header = json.loads(_b64url_decode(h_b64))
        if not isinstance(header, dict) or header.get("alg") != "HS256":  # blocks alg=none
            return None
        expected = hmac.new(secret, f"{h_b64}.{p_b64}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _b64url_decode(s_b64)):
            return None
        claims = json.loads(_b64url_decode(p_b64))
        if not isinstance(claims, dict):
            return None
        exp = claims.get("exp")
        if exp is not None and (not isinstance(exp, (int, float)) or now >= exp):
            return None
        return claims
    except (ValueError, TypeError, binascii.Error):
        return None


def role_from_claims(claims: dict[str, Any]) -> str:
    return "admin" if claims.get("role") == "admin" else "viewer"  # least privilege


# --------------------------------------------------------------------------- #
# Authorisation policy
# --------------------------------------------------------------------------- #
def is_admin_tool(name: str) -> bool:
    return name.strip().casefold().startswith(ADMIN_PREFIX)


def authorize_tool_call(role: str, tool_name: str) -> bool:
    return role == "admin" or not is_admin_tool(tool_name)


# --------------------------------------------------------------------------- #
# HTTP plumbing
# --------------------------------------------------------------------------- #
@dataclass
class Config:
    downstream_url: str
    secret: bytes
    filter_tools_list: bool = False
    downstream_timeout_s: float = 10.0


def rpc_error(rid: Any, code: int, message: str, data: Any = None) -> dict[str, Any]:
    err: dict[str, Any] = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": err}


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "mcp-gateway/1.0"

    @property
    def cfg(self) -> Config:
        return self.server.cfg  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args: Any) -> None:  # route access log to logging/stderr
        log.debug("%s - %s", self.address_string(), fmt % args)

    # ---- responses ---------------------------------------------------------
    def _send(self, status: int, body: bytes = b"", headers: dict[str, str] | None = None) -> None:
        self.send_response(status)
        for k, v in (headers or {}).items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if body:
            self.wfile.write(body)

    def _send_json(self, status: int, obj: Any, headers: dict[str, str] | None = None) -> None:
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        self._send(status, json.dumps(obj).encode(), h)

    # ---- request handling --------------------------------------------------
    def do_GET(self) -> None:
        self._method_not_allowed()

    do_PUT = do_DELETE = do_PATCH = do_GET

    def _method_not_allowed(self) -> None:
        self.close_connection = True
        self._send_json(405, {"error": "method_not_allowed"}, {"Allow": "POST"})

    def do_POST(self) -> None:
        # 1) read the body first so keep-alive connections never desync
        try:
            length = int(self.headers.get("Content-Length", ""))
        except ValueError:
            self.close_connection = True
            return self._send_json(411, {"error": "length_required"})
        if length < 0 or length > MAX_BODY:
            self.close_connection = True
            return self._send_json(413, {"error": "payload_too_large"})
        raw = self.rfile.read(length)

        if self.path.split("?", 1)[0] != "/mcp":
            return self._send_json(404, {"error": "not_found"})

        # 2) authenticate
        claims = self._authenticate()
        if claims is None:
            return self._send_json(
                401, {"error": "unauthorized"}, {"WWW-Authenticate": 'Bearer realm="mcp-gateway"'}
            )
        role = role_from_claims(claims)

        # 3) parse JSON-RPC envelope
        try:
            msg = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return self._send_json(400, rpc_error(None, PARSE_ERROR, "Parse error"))
        if isinstance(msg, list):
            return self._send_json(400, rpc_error(None, INVALID_REQUEST, "Batch requests are not supported"))
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0" or not isinstance(msg.get("method"), str):
            rid = msg.get("id") if isinstance(msg, dict) else None
            return self._send_json(400, rpc_error(rid if isinstance(rid, (str, int)) else None,
                                                  INVALID_REQUEST, "Invalid Request"))

        method: str = msg["method"]
        is_notification = "id" not in msg
        rid = msg.get("id")

        # 4) method-level policy
        if method == "tools/call":
            params = msg.get("params")
            name = params.get("name") if isinstance(params, dict) else None
            if not isinstance(name, str):
                if is_notification:
                    return self._send(202)
                return self._send_json(200, rpc_error(rid, INVALID_PARAMS, "Invalid params: params.name must be a string"))
            if not authorize_tool_call(role, name):
                log.warning("DENY role=%s tool=%r", role, name[:100])
                if is_notification:
                    return self._send(202)
                return self._send_json(200, rpc_error(rid, UNAUTHORIZED_TOOL_CALL, "Unauthorized Tool Call",
                                                      {"tool": name}))
            log.info("ALLOW role=%s tool=%r", role, name[:100])

        # 5) forward (original bytes)
        self._forward(raw, role, method, rid, is_notification)

    def _authenticate(self) -> dict[str, Any] | None:
        auth = self.headers.get("Authorization", "")
        scheme, _, token = auth.partition(" ")
        if scheme.lower() != "bearer" or not token.strip():
            return None
        return verify_token(token.strip(), self.cfg.secret)

    def _forward(self, raw: bytes, role: str, method: str, rid: Any, is_notification: bool) -> None:
        fwd_headers = {
            k: v for k, v in self.headers.items() if k.lower() not in HOP_BY_HOP
        }
        fwd_headers["Content-Type"] = "application/json"
        fwd_headers["X-Gateway-Role"] = role  # lets downstream audit; strip/ignore if untrusted
        req = urllib.request.Request(self.cfg.downstream_url, data=raw, headers=fwd_headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.cfg.downstream_timeout_s) as resp:
                status, body, rheaders = resp.status, resp.read(), resp.headers
        except urllib.error.HTTPError as e:  # downstream answered non-2xx: pass it through
            status, body, rheaders = e.code, e.read(), e.headers
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            log.error("downstream unavailable: %s", e)
            if is_notification:
                return self._send(502)
            return self._send_json(502, rpc_error(rid, DOWNSTREAM_UNAVAILABLE, "Downstream server unavailable"))

        if (
            method == "tools/list"
            and role != "admin"
            and self.cfg.filter_tools_list
            and "application/json" in (rheaders.get("Content-Type") or "")
        ):
            body = _filter_admin_tools(body)

        out_headers = {k: v for k, v in rheaders.items() if k.lower() in RESPONSE_HEADER_ALLOWLIST}
        self._send(status, body, out_headers)


def _filter_admin_tools(body: bytes) -> bytes:
    try:
        doc = json.loads(body)
        tools = doc["result"]["tools"]
        doc["result"]["tools"] = [t for t in tools if not is_admin_tool(str(t.get("name", "")))]
        return json.dumps(doc).encode()
    except (ValueError, KeyError, TypeError, AttributeError):
        return body  # not the shape we expected: pass through unchanged


def make_server(host: str, port: int, cfg: Config) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer((host, port), GatewayHandler)
    srv.daemon_threads = True
    srv.cfg = cfg  # type: ignore[attr-defined]
    return srv


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--downstream", default="http://127.0.0.1:9000/mcp")
    ap.add_argument("--filter-tools-list", action="store_true", help="hide admin_* tools from non-admins")
    args = ap.parse_args()

    secret = os.environ.get("GATEWAY_JWT_SECRET")
    if not secret:
        sys.exit("set GATEWAY_JWT_SECRET (HS256 signing secret)")
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, format="%(asctime)s %(levelname)s %(message)s")
    cfg = Config(args.downstream, secret.encode(), args.filter_tools_list)
    srv = make_server(args.host, args.port, cfg)
    log.info("gateway on http://%s:%d/mcp -> %s", args.host, srv.server_address[1], args.downstream)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
