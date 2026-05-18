from fastapi import Security, HTTPException
from fastapi.security import APIKeyHeader


api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(key: str = Security(api_key_header)):
    from config.settings import settings
    if not settings.API_KEY:
        return   # auth disabled in dev
    if key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
