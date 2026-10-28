"""Set up, or change, the hidden admin login. It is only for you, to debug and test the site.

  python scripts/set_admin.py                  # asks for a username and a password
  python scripts/set_admin.py --new-address    # also moves the login page to a new secret address

Writes these lines to .env (every other line stays as it is):
  PAPERSCOPE_ADMIN_USER            the username
  PAPERSCOPE_ADMIN_PASSWORD_HASH   the password, hashed with scrypt (the password itself is never stored)
  PAPERSCOPE_ADMIN_PATH            the secret address of the login page (made once, then kept)
  PAPERSCOPE_SECRET_KEY            signs the sign-in cookie (made once, then kept)
then prints the address to open. Restart the app afterwards. A new password signs out
every browser that was signed in. To switch the admin login off, delete these lines.
"""
import argparse
import getpass
import os
import secrets
import sys

from werkzeug.security import generate_password_hash

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MIN_LENGTH = 12


def read_env(path):
    out = {}
    try:
        with open(path, encoding="utf-8-sig") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip().strip('"').strip("'")
    except OSError:
        pass
    return out


def update_env(path, values):
    """Sets KEY=value lines in the .env file at path, keeping every other line as it is."""
    lines, newline = [], "\n"
    if os.path.exists(path):
        with open(path, encoding="utf-8-sig", newline="") as f:
            text = f.read()
        newline = "\r\n" if "\r\n" in text else "\n"
        lines = text.splitlines()
    left = dict(values)
    out = []
    for line in lines:
        k = line.split("=", 1)[0].strip() if "=" in line and not line.lstrip().startswith("#") else None
        out.append(f"{k}={left.pop(k)}" if k in left else line)
    if left:
        if out and out[-1].strip():
            out.append("")
        out.append("# Admin login (python scripts/set_admin.py). Keep these to yourself.")
        out += [f"{k}={v}" for k, v in left.items()]
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(newline.join(out) + newline)
    if os.name == "posix":
        os.chmod(path, 0o600)           # keys and the password hash: readable by you only


def admin_values(user, password, old, new_address=False):
    """The .env lines for this username and password, keeping the old address and cookie key."""
    return {"PAPERSCOPE_ADMIN_USER": user,
            "PAPERSCOPE_ADMIN_PASSWORD_HASH": generate_password_hash(password, method="scrypt"),
            "PAPERSCOPE_ADMIN_PATH": (not new_address and old.get("PAPERSCOPE_ADMIN_PATH")) or secrets.token_urlsafe(12),
            "PAPERSCOPE_SECRET_KEY": old.get("PAPERSCOPE_SECRET_KEY") or secrets.token_hex(32)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-address", action="store_true", help="move the login page to a new secret address")
    ap.add_argument("--env", default=os.path.join(HERE, ".env"), help="the .env file to write (default: the one next to app.py)")
    ap.add_argument("--site", help="the site's address, for the login address printed at the end "
                                   "(default: PAPERSCOPE_SITE in .env, else this computer)")
    args = ap.parse_args()
    old = read_env(args.env)

    user = input(f"Admin username{' [' + old['PAPERSCOPE_ADMIN_USER'] + ']' if old.get('PAPERSCOPE_ADMIN_USER') else ''}: ").strip() \
        or old.get("PAPERSCOPE_ADMIN_USER", "")
    if not user or any(c.isspace() for c in user):
        sys.exit("The username must be one word.")
    password = getpass.getpass(f"Admin password (at least {MIN_LENGTH} characters, not shown): ")
    if len(password) < MIN_LENGTH:
        sys.exit(f"The password is too short: use at least {MIN_LENGTH} characters.")
    if getpass.getpass("The same password again: ") != password:
        sys.exit("The two passwords are different. Nothing was changed.")

    values = admin_values(user, password, old, args.new_address)
    update_env(args.env, values)
    site = args.site or (f"https://{old['PAPERSCOPE_SITE']}" if old.get("PAPERSCOPE_SITE") else
                         f"http://127.0.0.1:{os.environ.get('PORT') or old.get('PORT') or '5000'}")
    print(f"\nSaved in {args.env}. Restart the app, then sign in at:\n\n"
          f"    {site.rstrip('/')}/{values['PAPERSCOPE_ADMIN_PATH']}/\n\n"
          "Bookmark it and keep it to yourself: nothing on the site links there.")


if __name__ == "__main__":
    main()
