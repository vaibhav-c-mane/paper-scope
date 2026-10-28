"""The hidden admin login: one account, only for the owner, to debug and test the site.

PaperScope is free for everyone and has no user accounts. Nothing on the public page
links here and none of this code is sent to visitors. The login page lives at a secret
address (PAPERSCOPE_ADMIN_PATH in .env) and checks a hashed password; without them the
admin part is switched off and every admin address answers 404. Set them up with

    python scripts/set_admin.py

Signed in, the search page also shows "How this search worked" (the engine's trace and
the test settings), the OpenAlex budget and why each paper ranked where it did, and the
API sends the trace and the ranking numbers that visitors never get.
"""
import hashlib
import hmac
import logging
import math
import os
import re
import secrets
import threading
import time
from datetime import timedelta

from flask import Blueprint, Response, abort, redirect, render_template, request, send_from_directory, session
from werkzeug.security import check_password_hash

from dsa.lru import LRUCache

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_TRIES = 5                   # wrong passwords in a row from one address before it has to wait
LOCK_S = 15 * 60                # how long it waits
SESSION_HOURS = 12              # a sign-in lasts this long
PATH_RE = re.compile(r"[A-Za-z0-9_-]{8,64}")
# Sign-ins and wrong passwords, with the address they came from (never the name or password typed),
# so the owner can see in the app's log if someone found the login page.
log = logging.getLogger("paperscope.admin")


def settings_from_env():
    """The admin login from .env, or None (switched off) when a part is missing."""
    s = {"path": os.environ.get("PAPERSCOPE_ADMIN_PATH", "").strip().strip("/"),
         "user": os.environ.get("PAPERSCOPE_ADMIN_USER", "").strip(),
         "password_hash": os.environ.get("PAPERSCOPE_ADMIN_PASSWORD_HASH", "").strip(),
         "secret_key": os.environ.get("PAPERSCOPE_SECRET_KEY", "").strip()}
    if not (PATH_RE.fullmatch(s["path"]) and s["path"] not in ("static", "api") and s["user"] and s["password_hash"]):
        return None
    return s


class LoginGuard:
    """Wrong passwords per address, in an LRU cache whose entries expire LOCK_S seconds after
    the last wrong try. After MAX_TRIES in a row that address waits until its entry expires.
    The cache holds at most 1,000 addresses, so a flood of them cannot fill memory."""

    def __init__(self, tries=MAX_TRIES, lock_s=LOCK_S, clock=time.monotonic):
        self.tries, self.lock_s, self.clock = tries, lock_s, clock
        self.failed = LRUCache(1000, ttl=lock_s, clock=clock)      # address -> (wrong tries, time of the last)
        self.lock = threading.Lock()

    def wait_s(self, addr):
        with self.lock:
            n, last = self.failed.get(addr, (0, 0.0))
        return max(0.0, last + self.lock_s - self.clock()) if n >= self.tries else 0.0

    def wrong(self, addr):
        with self.lock:
            n, _ = self.failed.get(addr, (0, 0.0))
            self.failed.put(addr, (n + 1, self.clock()))

    def right(self, addr):
        with self.lock:
            self.failed.pop(addr)


class AdminLogin:
    def __init__(self, app, settings):
        self.settings = settings
        if not settings:
            return
        self.base = f"/{settings['path']}/"
        # stored in the signed cookie; a new password changes it, which signs every browser out
        self.mark = hashlib.sha256(settings["password_hash"].encode()).hexdigest()[:24]
        # usernames are compared as hashes of one length, so the time taken cannot tell its length
        self.user_digest = hashlib.sha256(settings["user"].encode()).digest()
        self.guard = LoginGuard()
        app.config.update(SECRET_KEY=settings.get("secret_key") or secrets.token_hex(32),
                          SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Strict",
                          PERMANENT_SESSION_LIFETIME=timedelta(hours=SESSION_HOURS))
        bp = Blueprint("admin", __name__, template_folder=HERE)
        bp.add_url_rule("/", "login", self.login, methods=["GET", "POST"])
        bp.add_url_rule("/logout", "logout", self.logout, methods=["POST"])
        bp.add_url_rule("/admin.<ext>", "asset", self.asset)
        bp.after_request(self.private)
        app.register_blueprint(bp, url_prefix="/" + settings["path"])

    @property
    def enabled(self):
        return bool(self.settings)

    def signed_in(self):
        return self.enabled and session.get("admin") == self.mark

    def search_page(self, index_path):
        """The search page with the admin tools added. Never cached, so signing out removes them."""
        with open(index_path, encoding="utf-8") as f:
            html = f.read()
        html = html.replace("</head>", f'<link rel="stylesheet" href="{self.base}admin.css">\n</head>')
        html = html.replace("</body>", f'<script src="{self.base}admin.js"></script>\n</body>')
        r = Response(html, mimetype="text/html")
        r.headers["Cache-Control"] = "no-store"
        return r

    # ------------------------------------------------------------- pages
    def page(self, error=""):
        return render_template("login.html", base=self.base, error=error, signed_in=self.signed_in())

    def login(self):
        if request.method == "GET":
            return self.page()
        addr = request.remote_addr or "unknown"
        wait = self.guard.wait_s(addr)
        if wait:
            log.warning("admin sign-in refused: %s is waiting %d more min after %d wrong passwords",
                        addr, math.ceil(wait / 60), self.guard.tries)
            return self.page(f"Too many wrong tries. Try again in {math.ceil(wait / 60)} min."), 429
        try:        # checked even when the username is wrong, so a wrong name takes as long
            good = check_password_hash(self.settings["password_hash"], request.form.get("password", ""))
        except ValueError:                                  # not a hash: run scripts/set_admin.py
            good = False
        typed = hashlib.sha256(request.form.get("username", "").encode()).digest()
        if good and hmac.compare_digest(typed, self.user_digest):
            self.guard.right(addr)
            session.clear()
            session.permanent = True
            session["admin"] = self.mark
            log.warning("admin signed in from %s", addr)
            return redirect("/")
        self.guard.wrong(addr)
        log.warning("admin sign-in failed: wrong username or password from %s", addr)
        return self.page("Wrong username or password."), 401

    def logout(self):
        session.clear()
        return redirect("/")

    def asset(self, ext):
        if ext not in ("js", "css") or not self.signed_in():
            abort(404)
        return send_from_directory(HERE, "admin." + ext)

    @staticmethod
    def private(response):
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Robots-Tag"] = "noindex, nofollow"
        return response
