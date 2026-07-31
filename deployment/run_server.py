"""API 서버 실행 스크립트."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("deployment.api_server:app", host="0.0.0.0", port=8000, reload=True)
