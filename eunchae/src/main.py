import cv2
import numpy as np
from . import extract

from abc import ABC
from ultralytics import YOLO
from typing import Annotated, List
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile, Response

model = YOLO('./model/tft_detector.pt')

app = FastAPI()

@app.get('/')
async def root():
    return {'message': 'Hello World'}

@app.post('/predict/json/')
async def predict_json(image: UploadFile=File(...)):
    content = await image.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    static = extract.extract_static_content(img)
    dynamic = extract.extract_dynamic_content(img)

    return {**static, **dynamic}

@app.post('/predict/img/', responses={200: {'content': {'image/jpg': {}}}})
async def predict_image(image: UploadFile=File(...)):
    content = await image.read()
    nparr = np.fromstring(content, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.imencode('.jpg', img)[1].tostring()

    dynamic = extract.extract_dynamic_content(img)

    return Response(content=img, media_type="image/jpg")