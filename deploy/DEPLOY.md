# DFS v6 — ubuntserv deployment

Follows house patterns: compose in `/srv/compose/dfs/`, appdata in `/srv/appdata/dfs/`,
tunnel-only exposure behind Cloudflare Access, backups via `backup-home-configs.sh`.

## 1. Lay out appdata and migrate the working copy's state

```bash
sudo mkdir -p /srv/appdata/dfs/data /srv/compose/dfs
sudo cp ~/dfs-v6/.env /srv/appdata/dfs/.env && sudo chmod 600 /srv/appdata/dfs/.env
sudo cp -r ~/dfs-v6/data/distributions.json ~/dfs-v6/data/calibration_2025.json \
        ~/dfs-v6/data/results.db /srv/appdata/dfs/data/ 2>/dev/null
sudo git clone --branch v6-rebuild git@github.com:bel52/FanDuel_LeaguePicks.git \
        /srv/appdata/dfs/src-checkout 2>/dev/null || \
  sudo git -C /srv/appdata/dfs/src-checkout pull
sudo chown -R 998:998 /srv/appdata/dfs
```

## 2. Compose

```bash
sudo cp /srv/appdata/dfs/src-checkout/deploy/docker-compose.yml /srv/compose/dfs/
cd /srv/compose/dfs && sudo docker compose up -d --build
curl -s http://127.0.0.1:8093/health | python3 -m json.tool
```

Expect `"ok": true` and `"calibrated_distributions": true`. If calibrated is false,
the distributions didn't migrate — copy `data/distributions.json` again and restart.

## 3. Cloudflare tunnel + Access

Dashboard-managed connector (house pattern):
1. Zero Trust → Tunnels → the ubuntserv connector → Public Hostname → Add:
   `dfs.leathfam.com` → `http://localhost:8093`
2. Zero Trust → Access → Applications → Add: self-hosted, `dfs.leathfam.com`,
   policy = your email OTP group (same policy as other internal apps).
3. Verify the DNS record landed in the leathfam.com zone (known tunnel-CLI gotcha —
   dashboard flow is fine, but check).

```bash
curl -s -o /dev/null -w "%{http_code}\n" https://dfs.leathfam.com/health   # 302 to Access = correct
```

## 4. Updates (the new deploy path — replaces tarballs)

```bash
cd /srv/appdata/dfs/src-checkout && sudo git pull && \
  cd /srv/compose/dfs && sudo docker compose up -d --build
```

Data (`/srv/appdata/dfs/data`) is a bind mount and survives every rebuild — a code
update can no longer clobber the calibration or the results log.

## 5. Backups

Add to `backup-home-configs.sh` (Tedious tier — RAID1 + QNAP):
- `/srv/appdata/dfs/data`   (results.db, calibration, distributions — the season's record)
- `/srv/appdata/dfs/.env`   (keys)
- `/srv/compose/dfs/`

## 6. Not yet wired (next build)

- n8n cadence: Wed build reminder, Thu CSV nag, Sun 11:30 / 15:45 / 19:45 swap checks
  calling `POST /api/swap`, Pushover delivery of lineup + swap alerts.
- Key rotation before Week 1: FantasyPros + Odds keys (both exposed in chat) — new
  values only in `/srv/appdata/dfs/.env`.
