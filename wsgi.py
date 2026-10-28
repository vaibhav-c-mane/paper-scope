"""PaperScope for a public website: the same app as app.py, served by waitress (a production
web server that works on Linux and Windows) instead of Flask's development server.

  python wsgi.py          listens on every network address, port $PORT (default 8000)

On a host, set the start command to `python wsgi.py` (the Procfile says so for hosts that read
one), or point a host's own WSGI setting at `wsgi:application`. On your own server,
deploy/oracle-setup.sh sets all of this up (docs/FREE_HOSTING.md). Settings, as environment
variables on the host or in .env:
  PORT                 the port to listen on; most hosts set it themselves
  PAPERSCOPE_HOST      the network address to listen on (default 0.0.0.0, every one). 127.0.0.1
                       when a web server on the same machine (Caddy, nginx) passes requests on, so
                       nobody can reach the app without going through it.
  PAPERSCOPE_PROXIES   how many proxies pass requests on to the app (default 1: the host's).
                       The app finds each visitor's real address with it. 0 only when visitors
                       reach the app directly, with nothing in front of it.
  PAPERSCOPE_HTTPS     on (default): the sign-in cookie only travels over HTTPS and browsers are
                       told to always use HTTPS. off only to try this on your own computer over http.
  PAPERSCOPE_THREADS   requests served at the same time (default 8)

It runs as one process on purpose: the papers in memory, the OpenAlex budget and the admin
login's wrong-password counts are shared by all requests. Do not start several copies of it
(several gunicorn workers, for example) for one site.
"""
import logging
import os

from app import create_app, describe, for_deployment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def _setting(name, default):
    return (os.environ.get(name) or "").strip() or default


application = for_deployment(create_app(), proxies=int(_setting("PAPERSCOPE_PROXIES", "1")),
                             https=_setting("PAPERSCOPE_HTTPS", "on").lower() != "off")

if __name__ == "__main__":
    from waitress import serve

    logging.getLogger("paperscope").info(describe(application))
    # The proxy headers go through untouched: the app decides which entry to trust (app.for_deployment).
    # Requests are small (the largest is the admin sign-in form), so a bigger body is refused at once
    # instead of being stored (waitress would accept up to 1 GB).
    serve(application, host=_setting("PAPERSCOPE_HOST", "0.0.0.0"), port=int(_setting("PORT", "8000")),
          threads=int(_setting("PAPERSCOPE_THREADS", "8")), clear_untrusted_proxy_headers=False,
          max_request_body_size=64 * 1024)
