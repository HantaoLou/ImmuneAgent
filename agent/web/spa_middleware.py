import logging

from fastapi import HTTPException, Request
from fastapi.responses import FileResponse

from web.settings import settings

logger = logging.getLogger(__name__)


async def spa_middleware(request: Request, call_next):
    """Middleware to handle SPA routing by serving index.html for non-API routes"""

    # Skip if it's an API route or static file
    spa_paths = ["/auth", "/agents", "/chats", "/console", "/tools"]
    static_paths = ["/assets", "/index.html"]
    logger.info(f"request.url.path: {request.url.path}")
    if any(request.url.path.startswith(path) for path in spa_paths):
        logger.info(f"serving spa file: {request.url.path}")
        return FileResponse(f"{settings.frontend_path}/index.html")
    elif any(request.url.path.startswith(path) for path in static_paths):
        logger.info(f"serving static file: {request.url.path}")
        return FileResponse(f"{settings.frontend_path}{request.url.path}")
    response = await call_next(request)
    return response
