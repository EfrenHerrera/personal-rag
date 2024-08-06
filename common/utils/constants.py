import os, chromadb
from chromadb.config import Settings

# Define the Chroma settings
CHROMA_SETTINGS = chromadb.HttpClient(host="host.docker.internal", port=8000, settings=Settings(allow_reset=True, anonymized_telemetry=False))
