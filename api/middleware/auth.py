import hmac

from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    """Dependency enforcing optional API-key auth (P0.6).

    - settings.API_KEY empty  → auth disabled (local dev / demo mode).
    - settings.API_KEY set    → /query and /warmup require the X-API-Key
      header with the exact value. Comparison is constant-time.
    """
    from config.settings import settings
    if not settings.API_KEY:
        return   # auth disabled in dev
    if not key or not hmac.compare_digest(key, settings.API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
