import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

log = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        t0       = time.time()
        response = await call_next(request)
        elapsed  = round(time.time() - t0, 3)

        log.info(
            f"{request.method} {request.url.path} "
            f"-> {response.status_code}  ({elapsed}s)"
        )
        response.headers["X-Response-Time"] = str(elapsed)
        return response
