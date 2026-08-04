#!/usr/bin/env python3
"""Small launcher so the bash command line doesn't need --host/--port flags."""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("backend.main:app", **{"host": "0.0.0.0", "port": 8000, "log_level": "info"})
