"""HTTP route for serving generated audio files by path."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from fastapi import Depends, FastAPI, HTTPException, Request


def register_audio_route(
    app: FastAPI,
    verify_api_key: Callable[..., Any],
) -> None:
    """Register the ``GET /v1/audio`` route."""

    @app.get("/v1/audio")
    async def get_audio(path: str, request: Request, _: None = Depends(verify_api_key)):
        """Serve a generated audio file when path is within the allowed directory."""

        from fastapi.responses import FileResponse

        # Disallow absolute paths from the client outright.
        user_path = Path(path)
        if user_path.is_absolute():
            raise HTTPException(
                status_code=403, detail="Access denied: absolute paths are not allowed"
            )

        # Get allowed directory from app state.
        allowed_dir = Path(request.app.state.temp_audio_dir)

        # Construct the path and validate it's within allowed directory BEFORE resolving.
        # This prevents path traversal attacks (e.g., ../../etc/passwd).
        candidate_path = allowed_dir / user_path
        try:
            resolved_path = candidate_path.resolve(strict=True)
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Audio file not found")
        except (OSError, RuntimeError):
            raise HTTPException(status_code=400, detail="Invalid path")
        try:
            resolved_path.relative_to(allowed_dir)
        except ValueError:
            raise HTTPException(
                status_code=403, detail="Access denied: path outside allowed directory"
            )
        if not resolved_path.is_file():
            raise HTTPException(status_code=404, detail="Audio file not found")

        ext = resolved_path.suffix.lower()
        media_types = {
            ".mp3": "audio/mpeg",
            ".wav": "audio/wav",
            ".flac": "audio/flac",
            ".ogg": "audio/ogg",
        }
        return FileResponse(
            str(resolved_path), media_type=media_types.get(ext, "audio/mpeg")
        )
