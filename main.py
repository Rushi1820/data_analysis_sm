from fastapi import FastAPI
import router
import uvicorn

app = FastAPI()


app.include_router(router.router, prefix="/api/v1", tags=["AIinsights"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="debug")
