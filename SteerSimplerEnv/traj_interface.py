#!/usr/bin/env python3
"""Remote trajectory annotation via ngrok (fixed Flask send_file syntax).

Usage remains the same as before. The `/image/current` route now correctly
handles Flask 2.x vs 3.x keyword differences without generating a syntax
error.
"""
from __future__ import annotations

import os
import queue
import threading
import time
import inspect
from pathlib import Path
from typing import List, Optional

import cv2 as cv
import numpy as np
from flask import Flask, jsonify, request, send_file
from pyngrok import ngrok
import os, threading, queue
from pathlib import Path
from flask import Flask
from pyngrok import ngrok
from pyngrok.exception import PyngrokNgrokHTTPError


_STATIC_DIR = Path(__file__).with_suffix("").parent / "_static"
_HTML_PAGE = _STATIC_DIR / "index.html"

_DEFAULT_HTML = r"""<!DOCTYPE html>
<html lang=\"en\"><head><meta charset=\"utf-8\"><title>Trajectory</title></head><body>Simple placeholder</body></html>"""


class RemoteTrajServer:
    def __init__(
        self,
        port: int = 5001,
        authtoken: str | None = None,
        *,
        proto: str = "http",                     # "http" or "tcp"
        domain: str | None = None,               # e.g., "my-simpler.ngrok.app" (HTTP/TLS)
        remote_addr: str | None = None           # e.g., "2.tcp.ngrok.io:12345" (TCP)
    ):
        # ---- ngrok auth / config ----
        if authtoken is None:
            authtoken = os.getenv("NGROK_AUTHTOKEN")
        if authtoken:
            ngrok.set_auth_token(authtoken)

        # Prefer explicit args; fall back to env vars
        domain = domain or os.getenv("NGROK_DOMAIN")
        remote_addr = remote_addr or os.getenv("NGROK_REMOTE_ADDR")

        # ---- app + worker thread ----
        self._q: "queue.Queue[list[list[int]] | None]" = queue.Queue(maxsize=1)
        self._latest_frame: Path = Path("/tmp/traj_frame.png")

        self._app = Flask(__name__)
        self._register_routes()

        self._thread = threading.Thread(
            target=self._app.run, kwargs={"host": "0.0.0.0", "port": port}, daemon=True
        )
        self._thread.start()

        # ---- start tunnel (Pay-as-you-go requires reserved endpoint) ----
        try:
            if proto == "http":
                if not domain:
                    raise RuntimeError(
                        "ngrok Pay-as-you-go requires a reserved domain for HTTP/TLS. "
                        "Set NGROK_DOMAIN or pass domain=..."
                    )
                # self._tunnel = ngrok.connect(addr=port, proto="http", domain=domain)
                self._tunnel = ngrok.connect(addr=port, proto="http")
            elif proto == "tcp":
                if not remote_addr:
                    raise RuntimeError(
                        "ngrok Pay-as-you-go requires a reserved TCP address. "
                        "Set NGROK_REMOTE_ADDR or pass remote_addr=..."
                    )
                self._tunnel = ngrok.connect(addr=port, proto="tcp", remote_addr=remote_addr)
            else:
                raise ValueError(f"Unsupported proto={proto!r}; use 'http' or 'tcp'.")

            self._public_url = self._tunnel.public_url
            print(f"[traj-remote] Public URL: {self._public_url}")

        except PyngrokNgrokHTTPError as e:
            if "ERR_NGROK_15002" in str(e):
                raise RuntimeError(
                    "ngrok requires a reserved domain (HTTP/TLS) or reserved TCP address on your plan. "
                    "Reserve one and set NGROK_DOMAIN or NGROK_REMOTE_ADDR."
                ) from e
            raise

    # -------------------------------------------------
    def public_url(self) -> str:  # noqa: D401 (imperative)
        return self._public_url

    def request_traj(self, img: np.ndarray, timeout: int = 120) -> Optional[List[List[int]]]:
        cv.imwrite(str(self._latest_frame), cv.cvtColor(img, cv.COLOR_RGB2BGR))
        while not self._q.empty():
            try:
                self._q.get_nowait()
            except queue.Empty:
                break
        print(f"[traj-remote] Awaiting clicks at {self._public_url} (timeout {timeout}s)…")
        try:
            pts = self._q.get(timeout=timeout)
        except queue.Empty:
            return None
        return pts

    # -------------------------------------------------
    def _register_routes(self):
        app = self._app

        @app.route("/")
        def index():
            if not _HTML_PAGE.exists():
                _STATIC_DIR.mkdir(exist_ok=True)
                _HTML_PAGE.write_text(_DEFAULT_HTML, encoding="utf-8")
            return _HTML_PAGE.read_text(encoding="utf-8")

        @app.route("/image/current")
        def current_image():
            if not self._latest_frame.exists():
                return "No image yet", 404
            send_kwargs = {"mimetype": "image/png"}
            if "max_age" in inspect.signature(send_file).parameters:
                send_kwargs["max_age"] = 0
            else:
                send_kwargs["cache_timeout"] = 0
            return send_file(str(self._latest_frame), **send_kwargs)

        @app.route("/submit", methods=["POST"])
        def submit():
            data = request.get_json(force=True)
            pts = data.get("points")
            if not isinstance(pts, list):
                return jsonify({"error": "invalid payload"}), 400
            self._q.put(pts)
            return jsonify({"status": "ok"})

        @app.after_request
        def add_cors(resp):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
            return resp
