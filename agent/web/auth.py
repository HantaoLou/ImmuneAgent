import logging

from fastapi import HTTPException, Request, status
from fastapi.responses import JSONResponse

from web.settings import Settings

logger = logging.getLogger(__name__)
settings = Settings()


async def verify_token(request: Request):
    """Verify the access token from request headers"""
    # Skip auth for health check, OPTIONS requests, and static resources
    if (
        request.url.path == "/health"
        or request.method == "OPTIONS"
        or request.url.path.startswith("/assets/")
        or request.url.path.startswith("/auth")
        or request.url.path.endswith(
            (
                ".js",
                ".css",
                ".png",
                ".jpg",
                ".jpeg",
                ".gif",
                ".svg",
                ".ico",
                ".woff",
                ".woff2",
                ".ttf",
                ".eot",
            )
        )
        or request.url.path == "/"
        or request.url.path == "/index.html"
    ):
        return True

    # Get token from settings
    expected_token = settings.access_token

    # If no token is configured, allow all requests
    if not expected_token:
        return True

    # Get token from request headers
    auth_header = request.headers.get("Authorization")
    if not auth_header:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Check if it's a Bearer token
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authorization header format",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Extract token
    token = auth_header.split(" ")[1]

    # Verify token
    if token != expected_token:
        logger.warning(f"Invalid token attempt from {request.client.host}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid access token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return True


async def auth_middleware(request: Request, call_next):
    """Middleware to verify authentication before processing requests"""
    try:
        await verify_token(request)
        response = await call_next(request)
        return response
    except HTTPException as e:
        return JSONResponse(
            status_code=e.status_code,
            content={"detail": e.detail},
            headers=e.headers,
        )
