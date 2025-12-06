import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

from web.app import app
from web.auth import auth_middleware
from web.chat.api import router as chat_router
from web.context import set_origin
from web.session.api import router as session_router
from web.tools.api import router as tools_router
from web.session.api import usecase_router
from web.storage.api import router as oss_router
from web.settings import settings
from web.spa_middleware import spa_middleware

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Add authentication middleware
@app.middleware("http")
async def auth_middleware_wrapper(request, call_next):
    return await auth_middleware(request, call_next)


# Add SPA middleware for client-side routing
@app.middleware("http")
async def spa_middleware_wrapper(request, call_next):
    return await spa_middleware(request, call_next)


# Add HTTP origin middleware
@app.middleware("http")
async def http_origin_middleware(request, call_next):
    origin = request.headers.get("Origin")
    set_origin(origin)
    return await call_next(request)


@app.get(path="/health")
async def health_check():
    return {"healthy": True}


@app.get("/")
async def serve_index():
    """Serve the main index.html file"""
    return FileResponse("/opt/antibody_gen/dist/index.html")


# Mount static files from the frontend build directory
app.mount(
    "/assets", StaticFiles(directory="/opt/antibody_gen/dist/assets"), name="assets"
)

# Include API routes first
app.include_router(oss_router)  # OSS服务API，供MCP服务调用
app.include_router(session_router)
app.include_router(chat_router)
app.include_router(usecase_router)
app.include_router(tools_router)

if __name__ == "__main__":
    # Beautiful startup banner
    print("\n" + "=" * 80)
    print("🚀 ANTIBODY GENETICS AGENT SERVER")
    print("=" * 80)
    print(f"📍 Server URL: http://127.0.0.1:{settings.port}")
    print(f"🔑 Access Token: {settings.access_token}")
    print(f"🗄️  Database: {settings.database_url}")
    print(f"🌐 Frontend: {settings.frontend_path}")
    print("=" * 80)
    print("✨ Starting server...")
    print("=" * 80 + "\n")

    # Log the startup information
    logger.info("🚀 Starting Antibody Genetics Agent Server")
    logger.info(f"📍 Server will be available at: http://127.0.0.1:{settings.port}")
    logger.info(f"🔑 Access token: {settings.access_token}")
    logger.info(f"🗄️  Database: {settings.database_url}")

    uvicorn.run(app, host="0.0.0.0", port=settings.port)
