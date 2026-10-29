#!/usr/bin/env python3
"""Mock downstream MCP server: speaks just enough JSON-RPC over HTTP POST. Records every request it receives."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TOOLS = [
    {"name": "get_customer_record", "description": "Read a record", "inputSchema": {"type": "object"}},
    {"name": "admin_reset_key", "description": "Rotate an API key", "inputSchema": {"type": "object"}},
]


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        msg = json.loads(raw)
        self.server.calls.append({"message": msg, "headers": dict(self.headers.items())})
        if "id" not in msg:  # notification
            self.send_response(202)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        method = msg.get("method")
        if method == "tools/list":
            result = {"tools": TOOLS}
        elif method == "tools/call":
            result = {"content": [{"type": "text", "text": f"downstream executed {msg['params']['name']}"}], "isError": False}
        elif method == "initialize":
            result = {"protocolVersion": "2025-03-26", "capabilities": {"tools": {}}, "serverInfo": {"name": "mock", "version": "0"}}
        else:
            result = {}
        body = json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": result}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def make_server(port=0):
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    srv.daemon_threads = True
    srv.calls = []
    return srv


if __name__ == "__main__":
    s = make_server(9000)
    print("mock downstream on http://127.0.0.1:9000/mcp")
    s.serve_forever()
