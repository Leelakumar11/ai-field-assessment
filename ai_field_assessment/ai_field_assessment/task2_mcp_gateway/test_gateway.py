import json
import threading
import time
import unittest
import urllib.error
import urllib.request

import gateway
import mock_downstream

SECRET = b"test-secret"


def post(url, body, token=None, raw=False, headers=None):
    data = body if raw else json.dumps(body).encode()
    h = {"Content-Type": "application/json", **(headers or {})}
    if token:
        h["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, r.read(), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, e.read(), dict(e.headers)


def rpc(method, params=None, id=1):
    m = {"jsonrpc": "2.0", "method": method}
    if id is not None:
        m["id"] = id
    if params is not None:
        m["params"] = params
    return m


class GatewayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.down = mock_downstream.make_server()
        threading.Thread(target=cls.down.serve_forever, daemon=True).start()
        down_url = f"http://127.0.0.1:{cls.down.server_address[1]}/mcp"
        cls.gw = gateway.make_server("127.0.0.1", 0, gateway.Config(down_url, SECRET))
        threading.Thread(target=cls.gw.serve_forever, daemon=True).start()
        cls.url = f"http://127.0.0.1:{cls.gw.server_address[1]}/mcp"
        cls.admin = gateway.make_token(SECRET, "admin")
        cls.viewer = gateway.make_token(SECRET, "viewer")

    @classmethod
    def tearDownClass(cls):
        for s in (cls.gw, cls.down):
            s.shutdown(); s.server_close()

    def setUp(self):
        self.down.calls.clear()

    # ---- happy paths ----
    def test_tools_list_forwarded_transparently(self):
        status, body, _ = post(self.url, rpc("tools/list"), self.viewer)
        self.assertEqual(status, 200)
        self.assertEqual(len(json.loads(body)["result"]["tools"]), 2)
        self.assertEqual(self.down.calls[0]["message"]["method"], "tools/list")

    def test_viewer_can_call_regular_tool(self):
        status, body, _ = post(self.url, rpc("tools/call", {"name": "get_customer_record", "arguments": {}}), self.viewer)
        self.assertEqual(status, 200)
        self.assertIn("downstream executed get_customer_record", body.decode())

    def test_admin_can_call_admin_tool(self):
        status, body, _ = post(self.url, rpc("tools/call", {"name": "admin_reset_key"}), self.admin)
        self.assertEqual(status, 200)
        self.assertIn("downstream executed admin_reset_key", body.decode())

    def test_forwarded_bytes_are_untouched_and_user_token_not_leaked(self):
        raw = b'{ "jsonrpc":"2.0",  "id": 7, "method":"ping" }'
        post(self.url, raw, self.viewer, raw=True)
        seen = self.down.calls[0]
        self.assertEqual(seen["message"]["id"], 7)
        self.assertNotIn("Authorization", seen["headers"])
        self.assertEqual(seen["headers"]["X-Gateway-Role"], "viewer")

    # ---- authorisation ----
    def test_viewer_denied_admin_tool_without_touching_downstream(self):
        status, body, _ = post(self.url, rpc("tools/call", {"name": "admin_reset_key"}, id=42), self.viewer)
        doc = json.loads(body)
        self.assertEqual(status, 200)
        self.assertEqual(doc["error"]["code"], -32001)
        self.assertEqual(doc["error"]["message"], "Unauthorized Tool Call")
        self.assertEqual(doc["id"], 42)
        self.assertEqual(self.down.calls, [])

    def test_string_id_is_echoed(self):
        _, body, _ = post(self.url, rpc("tools/call", {"name": "admin_x"}, id="abc-1"), self.viewer)
        self.assertEqual(json.loads(body)["id"], "abc-1")

    def test_case_and_whitespace_variants_denied(self):
        for name in ["ADMIN_reset_key", "Admin_reset_key", " admin_reset_key"]:
            with self.subTest(name=name):
                _, body, _ = post(self.url, rpc("tools/call", {"name": name}), self.viewer)
                self.assertEqual(json.loads(body)["error"]["code"], -32001)
        self.assertEqual(self.down.calls, [])

    def test_unknown_role_is_least_privilege(self):
        tok = gateway.make_token(SECRET, "superuser")
        _, body, _ = post(self.url, rpc("tools/call", {"name": "admin_reset_key"}), tok)
        self.assertEqual(json.loads(body)["error"]["code"], -32001)

    def test_denied_notification_gets_202_and_is_not_forwarded(self):
        status, body, _ = post(self.url, rpc("tools/call", {"name": "admin_reset_key"}, id=None), self.viewer)
        self.assertEqual((status, body), (202, b""))
        self.assertEqual(self.down.calls, [])

    def test_batch_rejected_so_it_cannot_smuggle_admin_calls(self):
        batch = [rpc("tools/list", id=1), rpc("tools/call", {"name": "admin_reset_key"}, id=2)]
        status, body, _ = post(self.url, batch, self.viewer)
        self.assertEqual(status, 400)
        self.assertEqual(json.loads(body)["error"]["code"], -32600)
        self.assertEqual(self.down.calls, [])

    # ---- authentication ----
    def test_missing_token(self):
        status, _, h = post(self.url, rpc("tools/list"))
        self.assertEqual(status, 401)
        self.assertIn("Bearer", h.get("WWW-Authenticate", ""))
        self.assertEqual(self.down.calls, [])

    def test_bad_signature_expired_and_alg_none(self):
        forged = gateway.make_token(b"other-secret", "admin")
        expired = gateway.make_token(SECRET, "admin", ttl_s=-10)
        h = gateway._b64url_encode(b'{"alg":"none"}')
        p = gateway._b64url_encode(b'{"role":"admin"}')
        for tok in [forged, expired, f"{h}.{p}.", "garbage", "a.b.c"]:
            with self.subTest(tok=tok[:20]):
                status, _, _ = post(self.url, rpc("tools/list"), tok)
                self.assertEqual(status, 401)
        self.assertEqual(self.down.calls, [])

    # ---- wire format ----
    def test_malformed_json(self):
        status, body, _ = post(self.url, b"{not json", self.viewer, raw=True)
        self.assertEqual(status, 400)
        doc = json.loads(body)
        self.assertEqual(doc["error"]["code"], -32700)
        self.assertIsNone(doc["id"])

    def test_invalid_envelope(self):
        for bad in [{"jsonrpc": "1.0", "method": "x", "id": 1}, {"id": 1}, {"jsonrpc": "2.0", "method": 5, "id": 1}, 7]:
            with self.subTest(bad=bad):
                status, body, _ = post(self.url, bad, self.viewer)
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"]["code"], -32600)

    def test_tools_call_without_name(self):
        for params in [None, {}, {"name": 5}, "x"]:
            with self.subTest(params=params):
                _, body, _ = post(self.url, rpc("tools/call", params), self.viewer)
                self.assertEqual(json.loads(body)["error"]["code"], -32602)
        self.assertEqual(self.down.calls, [])

    def test_wrong_path_and_method(self):
        req = urllib.request.Request(self.url.replace("/mcp", "/nope"), data=b"{}", method="POST",
                                     headers={"Authorization": f"Bearer {self.viewer}"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(req)
        self.assertEqual(cm.exception.code, 404)
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(self.url)
        self.assertEqual(cm.exception.code, 405)

    # ---- resilience / options ----
    def test_downstream_down_returns_sanitised_502(self):
        gw = gateway.make_server("127.0.0.1", 0, gateway.Config("http://127.0.0.1:1/mcp", SECRET))
        threading.Thread(target=gw.serve_forever, daemon=True).start()
        try:
            status, body, _ = post(f"http://127.0.0.1:{gw.server_address[1]}/mcp", rpc("ping", id=5), self.viewer)
        finally:
            gw.shutdown(); gw.server_close()
        doc = json.loads(body)
        self.assertEqual(status, 502)
        self.assertEqual(doc["id"], 5)
        self.assertNotIn("Connection refused", body.decode())

    def test_optional_tools_list_filtering(self):
        down_url = f"http://127.0.0.1:{self.down.server_address[1]}/mcp"
        gw = gateway.make_server("127.0.0.1", 0, gateway.Config(down_url, SECRET, filter_tools_list=True))
        threading.Thread(target=gw.serve_forever, daemon=True).start()
        url = f"http://127.0.0.1:{gw.server_address[1]}/mcp"
        try:
            names = lambda tok: [t["name"] for t in json.loads(post(url, rpc("tools/list"), tok)[1])["result"]["tools"]]
            self.assertEqual(names(self.viewer), ["get_customer_record"])
            self.assertEqual(len(names(self.admin)), 2)
        finally:
            gw.shutdown(); gw.server_close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
