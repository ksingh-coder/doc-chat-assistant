"""Application launcher script"""

import os
import uvicorn

if __name__ == "__main__":
    # Disable reload in Docker (check for containerized environment)
    # Enable reload only in local development
    is_docker = os.path.exists("/.dockerenv") or os.getenv("DOCKER_ENV") == "true"
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=not is_docker,
        reload_dirs=["./app"] if not is_docker else None
    )
