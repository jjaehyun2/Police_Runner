"""관제실(Control Room) 대시보드 서버.

SUMO 데모 러너는 이 서버와 직접 결합하지 않는다. 러너가 매 스텝
`scripts/sumo_demo/out/` 아래 파일을 덮어쓰고, 이 서버는 요청이 올 때마다
그 파일을 다시 읽어 돌려준다(파일 폴링). 파일이 없으면 대기 상태를 준다.

세 종류를 나눠서 제공한다. 지도 기하(network)는 한 번만 받으면 되고
용량이 크며, 장면(scene)은 매 프레임 바뀌고, 패널 수치(state)는 작다.
분리해 두면 숫자만 필요한 화면이 지도까지 내려받지 않는다.

실행:
    py -3.12 server.py --port 8020
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR.parent / "out"
STATE_PATH = OUT_DIR / "state.json"
NETWORK_PATH = OUT_DIR / "network.json"
SCENE_PATH = OUT_DIR / "scene.json"
INDEX_PATH = BASE_DIR / "index.html"
MAP_JS_PATH = BASE_DIR / "map.js"

WAITING_STATE = {"status": "waiting"}

#: 화면 코드는 개발 중 계속 바뀌는데 브라우저는 index.html/map.js 를 캐시한다.
#: 그러면 기능을 추가해도 사용자 화면에는 옛 버전이 계속 뜨고, "구현이 안 된
#: 것"과 구분이 되지 않는다. 로컬 데모 서버이므로 캐시를 끄는 편이 안전하다.
NO_CACHE = {
    "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
    "Pragma": "no-cache",
    "Expires": "0",
}

app = FastAPI(title="관제실 대시보드")


def _read_json(path: Path, fallback: dict) -> JSONResponse:
    """파일을 읽어 JSON으로 돌려준다. 실패하면 fallback.

    러너가 파일을 쓰는 도중에 읽으면 일시적으로 깨진 JSON이 나올 수 있다.
    대시보드가 죽는 것보다 한 프레임 건너뛰는 편이 낫다.
    """
    if not path.exists():
        return JSONResponse(fallback, headers=NO_CACHE)
    try:
        return JSONResponse(json.loads(path.read_text(encoding="utf-8")), headers=NO_CACHE)
    except (OSError, json.JSONDecodeError):
        return JSONResponse(fallback, headers=NO_CACHE)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INDEX_PATH, media_type="text/html", headers=NO_CACHE)


@app.get("/map.js")
def map_js() -> FileResponse:
    return FileResponse(MAP_JS_PATH, media_type="application/javascript", headers=NO_CACHE)


@app.get("/api/state")
def get_state() -> JSONResponse:
    return _read_json(STATE_PATH, WAITING_STATE)


@app.get("/api/network")
def get_network() -> JSONResponse:
    return _read_json(NETWORK_PATH, {})


@app.get("/api/scene")
def get_scene() -> JSONResponse:
    return _read_json(SCENE_PATH, {})


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="관제실 대시보드 서버")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8020)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":
    main()