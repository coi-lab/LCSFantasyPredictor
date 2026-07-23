"""
Lightweight web server to serve the LCS Fantasy Weekly Dashboard.
Run this script to view the dashboard in your web browser.
"""

import http.server
import os
import socketserver
import webbrowser

PORT = 8050
DASHBOARD_DIR = os.path.dirname(os.path.abspath(__file__))


class DashboardHandler(http.server.SimpleHTTPRequestHandler):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DASHBOARD_DIR, **kwargs)

    def end_headers(self):
        # This is a local development dashboard. Always serve current files so
        # browser caches cannot retain an older app.js after code changes.
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def run_server():
    print(f"🚀 Starting LCS Fantasy Weekly Dashboard at http://localhost:{PORT}")
    print("Press Ctrl+C to stop the server.\n")

    with socketserver.TCPServer(("", PORT), DashboardHandler) as httpd:
        try:
            # Auto-open browser
            webbrowser.open(f"http://localhost:{PORT}")
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


if __name__ == "__main__":
    run_server()
