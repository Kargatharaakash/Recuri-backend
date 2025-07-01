import os
import time
import requests

# Set this to your Render backend URL
RENDER_URL = os.getenv("KEEPALIVE_URL") or "https://recuri-backend.onrender.com/api/health"

def keepalive():
    while True:
        try:
            print(f"Pinging {RENDER_URL} ...")
            r = requests.get(RENDER_URL, timeout=10)
            print(f"Status: {r.status_code}")
        except Exception as e:
            print(f"Keepalive ping failed: {e}")
        time.sleep(600)  # 10 minutes

if __name__ == "__main__":
    keepalive()
