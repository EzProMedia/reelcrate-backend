# Reelcrate backend

FastAPI DJ-set analysis and vertical-video rendering service. Production uses one worker and a persistent Railway volume mounted at `/data`.

## Configuration

Set a private random `JWT_SECRET` of at least 32 bytes. Startup refuses missing, short, or public-default secrets. Preserve the existing valid production value to keep users signed in.

Admin routes require a separate `ADMIN_TOKEN` of at least 32 bytes, passed as `Authorization: Bearer …`. URL passwords and JWT-secret fallback are removed. Missing admin configuration disables admin routes only.

Keep `STRIPE_SECRET_KEY`, `STRIPE_WEBHOOK_SECRET`, `STRIPE_PRICE_ID`, `STRIPE_PRICE_ID_YEAR`, `APP_URL`, `RESEND_API_KEY`, and `RESEND_FROM` configured in Railway. Webhooks refuse unsigned events if the webhook secret is missing. No pricing changes are included.

## September fixes

- Applied the September patch, then strengthened its admin, email, and discount handling.
- Email outbox persists on the data volume in SQLite, retries delivery failures, and clears sensitive payloads after success. Delivery is at least once: a crash after the provider accepts an email can still cause a duplicate. Run one worker.
- Account writes are atomic, flushed, and preserve the previous valid file as `users.json.bak`. Corrupt account data fails visibly instead of becoming an empty account database. The backup is on the same volume; retain independent Railway backups for disaster recovery.
- Persistent per-account and per-client attempt limits protect authentication routes.
- Stripe network calls and clip-source extraction run off the event loop.
- Job status requires its owner’s login; returned clip links carry scoped, six-hour tickets. Downloads support byte ranges and attachment filenames for phone playback and saving.
- Cleanup runs periodically and skips unfinished jobs. One upload processes at a time to protect the small volume. Interrupted jobs become retryable failures after restart.
- Checkout accepts an optional promo, looks it up, and uses either an applied discount or manual entry. A discount-specific checkout rejection falls back to manual entry.

## Frontend

The current frontend is deployed separately to the existing Netlify project. Its update preserves promo codes across verification, authenticates job polling, adds a home-screen manifest and icons, and improves phone downloads. Publish both halves together. Existing open browser tabs should reload after the backend update.

An optional `frontend/` folder in a local checkout is served at `/app/` for local checks. The production backend Docker image remains API-only.

## Validation

Install the requirements plus pytest and httpx, then run `pytest -q test_regressions.py`. Tests isolate storage and mock payment/email services; they never charge or contact customers. Full rendering additionally requires ffmpeg and the configured fonts. Tests are not a four-hour workload benchmark or a live payment verification.

## Rollback

Restore the preceding GitHub commit and republish the prior Netlify deploy together. Keep the data volume intact. The JSON user format remains compatible; the old release ignores the additional outbox database and backup file.


## October 2026 clip-quality update (0.7.0)

Upload accepts `show_caption`, `framing=fit|crop`, optional DJ-name `watermark`, and an optional `logo` image (PNG/JPEG/WebP, at most 5 MB and 4096×4096 pixels). Logos are decoded and re-encoded, retain transparency, and persist through caption edits. Existing paid/trial logo restrictions still apply to the Reelcrate badge.

The analyzer uses local tempo with finer timing and periodicity confidence. BPM filters exclude uncertain and out-of-range candidates; half/double interpretations are reported consistently. Some ranges legitimately produce fewer clips. Do not describe inferred BPM as certified track metadata. Short or silent recordings cannot produce an arbitrary number of distinct clips.

The renderer preserves the full frame by default, offers a crop option, protects text from filter-syntax injection, keeps captions inside a safer posting area, and fixes invisible waveform and end-of-stream failures. Source audio remains unnormalized stereo; loudness and copyrighted music/platform rights are not changed.

The separate Netlify frontend supplies five animated visualizer previews made from a synthetic test beat, actual upload progress, custom-branding controls, last-job recovery and one private ZIP download. Placeholder stats, unsupported post scheduling and disconnected edit-style controls are removed.

Local verification uses isolated accounts and a 52-minute 1080p DJ recording; no demo media, customer records, credentials or test accounts belong in this repository. The unit suite covers known beat rates, noise confidence, chunk alignment, settings gates, logo validation, signed ZIP downloads and caption edits. Full render testing must additionally check output counts, durations, playback and audio. Actual iPhone/Android device testing and concurrent-user capacity testing remain separate from a desktop browser at phone size.
