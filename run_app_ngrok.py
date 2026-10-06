"""Start Cleave server and expose it via ngrok."""

import contextlib
import socket
import subprocess
import sys
import time


def is_port_in_use(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def kill_port(port: int):
    """Kill whatever is listening on the given port (Windows)."""
    try:
        result = subprocess.run(
            ["netstat", "-ano"],
            capture_output=True,
            text=True,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                pid = line.strip().split()[-1]
                print(f"  Killing PID {pid} on port {port}...")
                subprocess.run(["taskkill", "/PID", pid, "/F"], capture_output=True)
    except Exception as e:
        print(f"  Warning: {e}")


def main():
    port = 8321

    # 1. Free the port if occupied
    if is_port_in_use(port):
        print(f"Port {port} is in use, freeing it...")
        kill_port(port)
        time.sleep(2)

    # 2. Start the FastAPI server in a new window
    print(f"Starting Cleave server on port {port}...")
    server = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "cleave.app:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(port),
        ],
        creationflags=subprocess.CREATE_NEW_CONSOLE,
    )

    # 3. Wait for server to be ready
    print("Waiting for server to start...")
    for _ in range(15):
        if is_port_in_use(port):
            break
        time.sleep(1)
    else:
        print("ERROR: Server failed to start!")
        server.kill()
        return

    print(f"Server is running on http://localhost:{port}")

    # 4. Connect ngrok
    try:
        from pyngrok import conf, ngrok

        pyngrok_config = conf.get_default()
        pyngrok_config.auth_token = "3ACmVoldN0XsN3AO9TSe1RV1ER0_hkuXUPGWa1qi28qXP57p"

        print(f"Starting ngrok tunnel to port {port}...")
        tunnel = ngrok.connect(port)
        print(f"\n{'=' * 50}")
        print(f"  ngrok public URL: {tunnel.public_url}")
        print(f"{'=' * 50}\n")
        input("Press Enter to stop everything...\n")
    except Exception as e:
        print(f"ngrok error: {e}")
        input("Press Enter to stop the server...\n")
    finally:
        # Cleanup
        print("Shutting down...")
        with contextlib.suppress(Exception):
            ngrok.kill()
        server.kill()
        print("Done.")


if __name__ == "__main__":
    main()
