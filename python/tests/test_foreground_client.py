"""CPU-only checks for the optional persistent HTTP transport."""
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
import unittest
from unittest.mock import patch

from jetson_pi_foreground import ForegroundSession, ManagedForegroundSession


class TestForegroundClient(unittest.TestCase):
    def test_keepalive_reuses_connection_and_consumes_each_response(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                requests.append((self.client_address, self.path, body))
                data = json.dumps({"sequence": len(requests)}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        client = ForegroundSession(f"http://127.0.0.1:{server.server_port}/api", keep_alive=True)
        try:
            for index in range(5):
                result = client._request("POST", "/foreground/infer", {"index": index})
                self.assertEqual(result["sequence"], index + 1)
            self.assertEqual(len({row[0] for row in requests}), 1)
            self.assertEqual([row[1] for row in requests], ["/api/foreground/infer"] * 5)
            self.assertEqual([row[2]["index"] for row in requests], list(range(5)))
            client.close()
            client.close()
            self.assertIsNone(client._connection)
        finally:
            client.close()
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_ambiguous_failure_is_not_retried(self):
        client = ForegroundSession(keep_alive=True)
        with patch("jetson_pi_foreground.http.client.HTTPConnection") as connection:
            instance = connection.return_value
            instance.request.side_effect = http.client.RemoteDisconnected("lost response")
            with self.assertRaisesRegex(RuntimeError, "keep-alive request failed"):
                client._request("POST", "/foreground/infer", {"text": "A"})
            instance.request.assert_called_once()
            instance.close.assert_called_once()
            self.assertIsNone(client._connection)

    def test_existing_default_transport_remains_opt_out(self):
        client = ForegroundSession()
        self.assertFalse(client.keep_alive)
        client.close()

    def test_managed_server_inherits_optimization_environment(self):
        settings = {"PI_MODEL": "pi05", "PI0_PERSISTENT_CUDA_GRAPH": "1",
                    "PI0_DECODE_UNROLL": "10", "PI05_FP16_SHARED_CAST": "1",
                    "PI05_CACHE_TIME_MODULATION": "1", "LLAMA_ARG_THREADS": "8",
                    "LLAMA_ARG_CTX_SIZE": "131072"}
        with patch.dict("os.environ", settings), \
                patch("jetson_pi_foreground.Path.is_file", return_value=True), \
                patch.object(ManagedForegroundSession, "_wait_until_ready"), \
                patch("jetson_pi_foreground.subprocess.Popen") as popen:
            popen.return_value.poll.return_value = None
            client = ManagedForegroundSession(server_path="server", model_path="model.gguf",
                                              mmproj_path="vision.gguf", n_gpu_layers=99,
                                              gpu=0, keep_alive=True)
            try:
                env = popen.call_args.kwargs["env"]
                self.assertTrue(client.keep_alive)
                for key, value in settings.items():
                    self.assertEqual(env[key], value)
                self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            finally:
                client.close()
            popen.return_value.terminate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
