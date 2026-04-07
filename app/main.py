from fastapi import FastAPI, File, UploadFile, HTTPException
from .model import FoodModel
from PIL import Image
import io

app = FastAPI(title="Food Recognition API")

# Инициализируем модель один раз при запуске
model = FoodModel()

@app.post("/predict")
async def predict_food(file: UploadFile = File(...)):
    # Проверка формата файла
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Чтение изображения
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Инференс
        result = model.predict(image)
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ready"}