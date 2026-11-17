"""ASGI entry point.  Run locally with:  uvicorn app.main:app --reload"""

import logging

from app.factory import create_app

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
app = create_app()
