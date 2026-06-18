import http.server, urllib.request, threading, time

class TestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        is_api = self.path.startswith("/api/")
        print(f"  path={self.path!r} is_api={is_api}")
        if not is_api:
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

srv = http.server.HTTPServer(("", 18888), TestHandler)
t = threading.Thread(target=srv.serve_forever, daemon=True)
t.start()
time.sleep(0.3)

r = urllib.request.urlopen("http://localhost:18888/demo/index.html")
print("Cache-Control:", r.headers.get("Cache-Control"))
srv.shutdown()
