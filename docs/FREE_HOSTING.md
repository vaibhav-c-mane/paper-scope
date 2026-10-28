# Hosting PaperScope for free

PaperScope runs on a free Oracle Cloud server, at a free DuckDNS address, with free HTTPS
from Caddy. Nothing here costs money: no domain to buy, no monthly fee, no limit on views.

```
visitors → https://yourname.duckdns.org → Caddy (HTTPS) → PaperScope (python wsgi.py) on Oracle's free server
```

What you need: about an hour, a GitHub account, and a debit or credit card for Oracle's
identity check. Oracle does not charge the "Always Free" server, now or later; it may hold a
small amount for a moment to check the card, then releases it.

## 1. Put the newest code on GitHub

The server downloads PaperScope from your GitHub repository, including
`deploy/oracle-setup.sh`. On your computer, commit and push:

```bash
git add -A
git commit -m "Free hosting setup"
git push
```

## 2. Create a free Oracle Cloud account

1. Go to <https://www.oracle.com/cloud/free/> and click **Start for free**.
2. Enter your country, name and email, and confirm the email Oracle sends you.
3. Choose a password and a **Cloud Account Name** (any short name; you type it when you sign in).
4. Choose your **Home Region**, the one closest to most of your visitors. It cannot be
   changed later, and the free server must be in it.
5. Enter your address and phone, then the card for the identity check.
6. Wait for the email that says the account is ready (minutes, sometimes an hour), then
   sign in at <https://cloud.oracle.com>.

New accounts also get trial credit for 30 days. When it ends, the account simply stays on
Always Free. **Never click "Upgrade"**: everything below is Always Free, so nothing is ever charged.

## 3. Create the server

1. Open the menu (☰) → **Compute** → **Instances** → **Create instance**.
2. **Name**: `paperscope`.
3. **Image and shape** → **Edit**:
   * **Change image** → **Ubuntu** → **Canonical Ubuntu 24.04** → **Select image**.
   * **Change shape** → **Ampere** → **VM.Standard.A1.Flex** → set **OCPUs 1** and
     **Memory 2 GB** → **Select shape**. It is marked "Always Free-eligible".
   * A small server on purpose: Oracle may stop a free server whose CPU, network and memory
     all stay below 20% for a week, and PaperScope keeps 2 GB of memory above 20%.
4. **Networking**: keep **Create new virtual cloud network** and **Create new public subnet**,
   and keep **Automatically assign public IPv4 address** switched on.
5. **Add SSH keys**: choose **Generate a key pair for me** and click **Download private key**.
   Save it as `C:\Users\<you>\.ssh\paperscope.key`. It is the only key to the server: keep it,
   and never share it.
6. Leave the boot volume as it is and click **Create**.
7. When the status says **Running**, copy the **Public IP address** (four numbers like `152.67.1.23`).

If Oracle says **Out of capacity** for the Ampere shape, try another **Availability domain**
on the same page, try again later, or pick **VM.Standard.E2.1.Micro** (also Always Free,
slower, 1 GB); the setup script adds swap memory so it still works.

## 4. Open the web ports in Oracle's network

1. On the instance's page, under **Primary VNIC**, click the **Subnet** name.
2. Open **Security** (or **Security Lists**) → **Default Security List for …**.
3. **Add Ingress Rules**:
   * **Source CIDR**: `0.0.0.0/0`
   * **IP Protocol**: TCP
   * **Destination Port Range**: `80,443`
4. Click **Add Ingress Rules**. (The setup script opens the same ports in the server's own firewall.)

## 5. Get a free web address (DuckDNS)

1. Go to <https://www.duckdns.org> and sign in with GitHub or Google.
2. Under **domains**, type a name, for example `paperscope`, and click **add domain**.
   Your address is now `paperscope.duckdns.org`.
3. In that row's **current ip** box, paste the server's **Public IP address** and click **update ip**.
4. Keep the page open: the **token** shown at the top is needed for Google in step 9.

## 6. Connect to the server

On Windows, open **PowerShell** and run (with your server's IP):

```powershell
ssh -i "$env:USERPROFILE\.ssh\paperscope.key" ubuntu@152.67.1.23
```

Type `yes` the first time. You are now on the server: the prompt shows `ubuntu@paperscope`.

If SSH says **UNPROTECTED PRIVATE KEY FILE**, make the key readable only by you, then connect again:

```powershell
icacls "$env:USERPROFILE\.ssh\paperscope.key" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

## 7. Install PaperScope (on the server)

```bash
sudo apt-get update && sudo apt-get install -y git
git clone https://github.com/vaibhav-c-mane/paper-scope.git
bash paper-scope/deploy/oracle-setup.sh
```

The script installs everything, which takes 5 to 10 minutes. It asks for:

* your **OpenAlex API key**, **email** and **Semantic Scholar key** (Enter skips one). They are
  kept in `.env` on the server only;
* an **admin username and password** (at least 12 characters) for your hidden admin login;
* your **site address**, for example `paperscope.duckdns.org`.

At the end it prints your site's address and your secret admin login address. Save the admin
address somewhere private.

## 8. Check that it works

1. Open `https://paperscope.duckdns.org` and search for something. The first HTTPS visit can
   take a minute while Caddy gets the certificate.
2. Check the wrong-password lock sees real visitors: open your admin login address, sign in
   once with a **wrong** password, then on the server run:

   ```bash
   sudo journalctl -u paperscope -n 20
   ```

   The line `admin sign-in failed … from X` should show your own IP address (search Google
   for "what is my ip"). Then sign in with the right password.

## 9. Tell Google about the site

1. Go to <https://search.google.com/search-console> → **Add property** → **Domain** →
   `paperscope.duckdns.org` → **Continue**. Google shows a TXT record like
   `google-site-verification=abc123…`.
2. In a new browser tab, open this address with your DuckDNS name, your token (from step 5) and
   the whole value Google showed:

   ```
   https://www.duckdns.org/update?domains=paperscope&token=YOUR-TOKEN&txt=google-site-verification=abc123...&verbose=true
   ```

   It answers `OK`.
3. Back in Search Console, click **Verify**, then use **URL inspection** on your home page and
   click **Request indexing**.

## Keeping it running

| To | Do this on the server |
|---|---|
| Put new code online (after `git push` on your computer) | `bash ~/paper-scope/deploy/oracle-setup.sh` |
| See the site's log, live | `sudo journalctl -u paperscope -f` (Ctrl + C stops watching) |
| Change the admin password | `cd ~/paper-scope && .venv/bin/python scripts/set_admin.py && sudo systemctl restart paperscope` |
| Move the admin login page | the same, with `scripts/set_admin.py --new-address` |
| Restart the site | `sudo systemctl restart paperscope` |

Updates and restarts take the site away for a few seconds, too short to matter to visitors or
to Google. Ubuntu installs security updates by itself.

If the server's public IP ever changes (only if you delete and recreate the server), put the
new IP in DuckDNS's **current ip** box. If Oracle emails that it stopped an idle server, start
it again under **Compute** → **Instances** → **Start**.
