from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from .model import FoodModel
from PIL import Image
import io

app = FastAPI(title="Food Recognition API")

model = FoodModel()

@app.post("/predict")
async def predict_food(
    file: UploadFile = File(...),
    weight: float = Form(..., description="Weight of the dish in grams", gt=0)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        # Чтение изображения
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Инференс – получаем название блюда и КБЖУ на 100 г
        result = model.predict(image)
        dish = result["dish"]
        confidence = result["confidence"]
        nutrition_per_100g = result["nutrition_per_100g"]
        
        # Рассчитываем КБЖУ для указанного веса
        if isinstance(nutrition_per_100g, dict):
            multiplier = weight / 100.0
            nutrition_for_weight = {
                "calories": round(nutrition_per_100g["calories_per_100g"] * multiplier, 2),
                "protein": round(nutrition_per_100g["protein_per_100g"] * multiplier, 2),
                "carbs": round(nutrition_per_100g["carbs_per_100g"] * multiplier, 2),
                "fat": round(nutrition_per_100g["fat_per_100g"] * multiplier, 2),
                "fiber": round(nutrition_per_100g["fiber_per_100g"] * multiplier, 2)
            }
        else:
            # Если блюдо не найдено в базе
            nutrition_for_weight = "Cannot calculate for weight without nutrition data"
        
        return {
            "dish": dish,
            "confidence": confidence,
            "weight_g": weight,
            "nutrition_per_100g": nutrition_per_100g,
            "nutrition_for_weight": nutrition_for_weight
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "ready"}