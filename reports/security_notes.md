# Security & Privacy Notes — Known Limitations

**Healthcare RAG-Powered Medical Q&A Assistant**
**Written:** 2026-09-24 (Sept 2026 controlled repair, P2)

This document records the **actual current state** of security and privacy
handling. Items are labeled **[MITIGATED]** (a repair changed the behavior and
a test covers it), **[PARTIAL]**, or **[OPEN]** (documented only, not solved).
Nothing here should be read as a claim that an open item is fixed.

---

## 1. CORS wildcard — [OPEN]

- **State:** `config/settings.py` defines `CORS_ORIGINS: list = ["*"]`, and
  `api/main.py` passes it straight to `CORSMiddleware(allow_origins=...)`.
  The API therefore accepts browser requests from any origin.
- **Impact:** any website open in a user's browser can issue requests to the
  API (subject to the API-key check below). For a public demo this is
  convenient; for a restricted deployment it widens the attack surface.
- **Not done:** locking this down requires knowing the real deployment
  origin(s), which are not established. To restrict, set `CORS_ORIGINS` to an
  explicit allow-list (it is already env-overridable).
- **Note:** CORS is browser-enforced only; it is not an authentication
  mechanism and does not protect non-browser clients.

## 2. User-question logging — [OPEN]

- **State:** `api/routes/query.py` logs the first ~60–80 characters of every
  user question at INFO level (request receipt, cache HIT/MISS lines).
- **Impact:** questions submitted to a medical assistant can reveal health
  information. They are written to application logs in plaintext, alongside
  IP addresses from the web server, and inherit the retention/access policy
  of wherever logs are shipped (e.g., Azure App Service log streams).
- **Not done:** no redaction, no opt-out, no retention limit, and no
  documented log-handling policy. Reducing the log payload to the cache key
  hash plus status (no question text) is a small, safe change if desired;
  it was deliberately **not** made unilaterally here because the current
  logs are actively used for debugging.

## 3. Exception detail leakage — [MITIGATED]

- **State (before):** unhandled pipeline errors returned
  `detail=f"Pipeline error: {str(e)}"` to the client, potentially exposing
  internal paths/model names.
- **State (after, commit `ff493fc`):** the 500 response body is the generic
  string `Internal server error`; the full exception and traceback are logged
  server-side (`logger.error(..., exc_info=True)`).
- **Coverage:** no test asserts on the detail text; the change is intentionally
  minimal and visible in the diff.

## 4. Unauthenticated warmup — [MITIGATED]

- **State (before):** `GET /warmup` triggered full model loading with no
  authentication — a trivial resource-exhaustion vector on a public endpoint.
- **State (after, commit `92afb1a`):** `/warmup` requires the same API key as
  `/query` (`Depends(verify_api_key)`).
- **Residual:** `GET /health` remains intentionally unauthenticated so
  load-balancer probes work; it returns service/model status only.
- **Auth mechanism:** optional API key via the `X-API-Key` header, compared
  with `hmac.compare_digest` (constant time) in `api/middleware/auth.py`.
  When `settings.API_KEY` is empty, auth is disabled — acceptable for local
  development, **not** acceptable for any exposed deployment.
  In production via GitHub Actions, set the repository secret `API_KEY`
  (`.github/workflows/azure-deploy.yml` passes it through as an environment
  variable; no secret is committed). **Gap:** the Docker Compose production
  file (`docker/docker-compose.prod.yml`) does not currently pass `API_KEY`
  into the container environment — anyone deploying via that file must add
  it explicitly.

## 5. Prompt injection — [OPEN — documented only]

- **State:** text retrieved from the index is inserted directly into the LLM
  prompt. There is **no** injection sanitization, no delimiters/trust
  separation beyond simple formatting, and no output filtering. The grounding
  policy (P0.3) constrains *answers* to retrieved evidence but does not
  attempt to detect adversarial instructions inside that evidence.
- **Impact:** a record in the corpus containing instructions ("ignore
  previous directions…") could steer the generator. The corpus is a curated
  medical QA dataset, which lowers the likelihood of adversarial content, but
  this is a mitigation by provenance, not a control.
- **Not done:** any countermeasure. If attempted later, candidates include
  structured evidence delimiting, instruction-detection on retrieved chunks,
  and refusing answers whose evidence contains instruction-like content —
  each needs its own evaluation before being trusted.

## 6. Third-party Arabic translation API — [OPEN]

- **State:** the dashboard (`dashboard/index.html`) translates UI strings by
  calling `https://api.mymemory.translated.net/get` directly from the browser
  with an 8-second timeout. No API key, no backend proxy.
- **Impact:**
  - UI text (interface labels, not user questions) is sent to a third party.
  - Availability of the Arabic UI depends on a free, unversioned third-party
    service; there is no SLA and no fallback translation.
  - The request is visible to network observers (plain HTTPS to a third
    party); the service could change terms or behavior at any time.
- **Not done:** self-hosting translations, a bundled string table, or a
  backend proxy. The English UI does not depend on the service.

## 7. Other observations (informational)

- The nginx dashboard config (`docker/nginx-dashboard.conf`) sets
  `X-Content-Type-Options` and `X-Frame-Options` but no
  `Content-Security-Policy`; the SPA loads no third-party scripts, which
  limits (but does not formalize) script-injection risk.
- No secrets are committed; `.env.example` documents required variables and
  the deployment secret is expected as a GitHub Actions secret.

---

**Summary:** of the five P2 areas, exception-detail leakage and unauthenticated
warmup are fixed with the repairs; CORS wildcard, user-question logging,
prompt-injection handling, and the third-party translation dependency remain
**open and documented** — they require product/deployment decisions that were
out of scope for this repair.
