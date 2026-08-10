"""관제실(Control Room) 대시보드 서버.

SUMO 데모 러너는 이 서버와 직접 결합하지 않는다. 대신 매 스텝마다
`scripts/sumo_demo/out/state.json` 파일을 덮어써서 상태를 공유하고,
이 서버는 `/api/state` 요청이 들어올 때마다 그 파일을 다시 읽어 그대로
돌려준다(파일 폴링 방식). 파일이 없으면 대기 상태를 반환한다.

실행:
    py -3.12 server.py --port 8020
    또는
    py -3.12 -m uvicorn server:app --port 8020
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
STATE_PATH = BASE_DIR.parent / "out" / "state.json"
INDEX_PATH = BASE_DIR / "index.html"

WAITING_STATE = {"status": "waiting"}

app = FastAPI(title="관제실 대시보드")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_PATH, media_type="text/html")


@app.get("/api/state")
def get_state() -> JSONResponse:
    if not STATE_PATH.exists():
        return JSONResponse(WAITING_STATE)
    try:
        raw = STATE_PATH.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        # 러너가 파일을 쓰는 도중에 읽으면 일시적으로 깨진 JSON일 수 있다.
        # 실패해도 대시보드가 죽지 않도록 대기 상태로 대체한다.
        return JSONResponse(WAITING_STATE)
    return JSONResponse(data)


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="관제실 대시보드 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
