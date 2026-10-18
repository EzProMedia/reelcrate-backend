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

### Foreground framing model

Follow-the-DJ framing uses the small OpenCV YuNet 2023 model, detecting faces
four times per second and checking every frame for camera cuts. It follows a
prominent foreground face with a smoothed crop, restarting on shot changes.
Pixel-motion tracking bridges brief face occlusions; smaller crowd faces cannot
replace the foreground subject just because the DJ looks down.
Long detection gaps, ambiguous framing, or a face too large for the vertical
window use a full-frame fallback with an explicit result warning. This is the
phone UI default; Keep full frame is available for groups and showing the decks.

Automatic tags describe energy only. Spectral flux cannot verify a track blend,
drop, build-up, or audience reaction; the app no longer asserts those events.
Clip selection includes 40% of the chosen length before the audio peak.
The detector runs inside this service and does not send frames to another API.

Model: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet

The bundled `face_detection_yunet_2023mar.onnx` is distributed under the MIT
license included in `YUNET-LICENSE.txt`. SHA-256:
`8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.

## Music transitions and signup count (0.8.0)

Uploads accept `selection_mode=transitions|energy`. Transition mode compares
sustained harmonic and timbral changes, rejects recurring sections and short
returns, and retains music on both sides of a candidate handover. It requires
30 or 60 seconds. This is an estimate, not track fingerprinting: it can miss
mixes or confuse musical sections. Fewer qualified, nonoverlapping transitions
produce fewer clips with an explicit warning; energy peaks never pad the pack.
BPM filters require confident estimates on both sides. An uncertain or changing
tempo is not shown as one authoritative BPM.

The landing page reads only aggregate counts from `/api/founder-spots`.
The verified `ADMIN_EMAIL` account can review `/api/admin/signups` and import
existing Netlify records through the separate frontend `app/signups.html`.
Imports preserve the earliest signup date and do not send email or create
accounts. Counts deduplicate normalized emails across the waitlist and verified
accounts, exclude internal/test addresses, and never show negative remaining
spots. Contact records remain private on the persistent data volume. This does
not change Stripe promotion eligibility or coupon redemption limits.

Run `pytest -q test_regressions.py test_transitions_and_signups.py` for isolated
regression tests. Real-recording QA and private media belong outside this public
repository.


## Crowd-facing recordings (0.9.0)

`selection_mode=crowd` measures sustained, distributed visual activity after compensating camera translation, rotation and zoom. It accepts 15/30/60-second UI lengths, keeps context before/after each peak, avoids overlap, applies the existing confidence-based BPM filters, and never pads results with audio peaks. This is visual activity estimation on user-designated audience footage, not semantic cheering, emotion or song recognition. Lighting, low detail, rapid camera movement and animated graphics can affect results.

Optional `crowd_region` is a JSON array `[x,y,width,height]` of normalized coordinates. The phone UI lets the user draw this region on a local video preview, useful for livestreams containing animated posters. Validate finite, in-frame coordinates and a minimum 15% width/height before processing. Use the same region for analysis, output and caption edits. Crowd exports preserve the entire chosen region on a clean black background; face tracking is disabled even if requested through the API. All visual analysis stays within this service. No third-party vision API or new dependency is used.

Run `pytest -q test_regressions.py test_transitions_and_signups.py test_crowd.py`. User footage and generated review clips stay outside the public repository. The existing 2 GB upload limit still applies.
