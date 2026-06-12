"""
Token server — serves /api/token for the Vite frontend.

Run alongside the agent:
  uv run python token_server.py      (port 8080)
  uv run python main.py dev          (port 8081)
"""

import os
import uuid

from aiohttp import web
from dotenv import load_dotenv
from livekit.api import AccessToken, VideoGrants

load_dotenv()


async def token_handler(request: web.Request) -> web.Response:
    room_name = f"call-{uuid.uuid4().hex[:8]}"
    identity = f"caller-{uuid.uuid4().hex[:6]}"

    token = (
        AccessToken(os.environ["LIVEKIT_API_KEY"], os.environ["LIVEKIT_API_SECRET"])
        .with_identity(identity)
        .with_name("Caller")
        .with_grants(
            VideoGrants(
                room_join=True,
                room=room_name,
                can_publish=True,
                can_subscribe=True,
            )
        )
    )

    return web.json_response(
        {
            "token": token.to_jwt(),
            "url": os.environ["LIVEKIT_URL"],
            "room": room_name,
        },
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        },
    )


async def options_handler(request: web.Request) -> web.Response:
    return web.Response(
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
        }
    )


app = web.Application()
app.router.add_get("/api/token", token_handler)
app.router.add_options("/api/token", options_handler)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    web.run_app(app, host="0.0.0.0", port=port)
