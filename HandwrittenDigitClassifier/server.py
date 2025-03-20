from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import pickle, configparser
import uvicorn, argparse
from urllib.parse import urlparse
from dense_neural_class import *

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "API is working!"}

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Define request schema
class ImageData(BaseModel):
    image_vector: list

@app.post("/predict")
async def predict(data: ImageData):
    try:
        image_array = np.array(data.image_vector).reshape(1,-1)
        result = model.predict(image_array)[0]
        return {"digit": int(result)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Read API URL from config.ini
# config = configparser.ConfigParser()
# config.read("config.ini")
# API_URL = config.get("API", "url", fallback="http://0.0.0.0:5000/predict")
# PORT = int(config.get("API", "port", fallback=5000))

# uvicorn.run(app, host="localhost", port=PORT)

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run FastAPI server with configurable host and port.")
    parser.add_argument("--url", type=str, required=True, help="API URL to use for predictions (e.g., http://0.0.0.0:5000/predict)")

    # Parse arguments
    args = parser.parse_args()

    # Parse URL to extract host and port
    parsed_url = urlparse(args.url)
    host = parsed_url.hostname or "127.0.0.1"  # Default to allow all connections
    port = int(parsed_url.port) if parsed_url.port else 5000  # Default port if missing
    uvicorn.run("server:app", host=host, port=port)