import os
import json
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor

class FoodModel:
    def __init__(self):
        # Путь к папке с тремя файлами
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.weights_path = os.path.join(current_dir, "..", "models")
        
        print(f"Загрузка модели из локальной папки: {self.weights_path}")
        
        # Загружаем процессор и модель из локальной директории
        self.processor = ViTImageProcessor.from_pretrained(self.weights_path)
        self.model = ViTForImageClassification.from_pretrained(self.weights_path)
        self.model.eval()

        # Загружаем ваш справочник КБЖУ
        json_path = os.path.join(current_dir, "nutritional_database.json")
        with open(json_path, "r", encoding="utf-8") as f:
            self.nutrition_db = json.load(f)

    def predict(self, image: Image.Image):
        # Препроцессинг
        inputs = self.processor(images=image, return_tensors="pt")

        # Инференс
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()

        # Получаем текстовую метку из config.json модели
        label = self.model.config.id2label[predicted_class_idx]
        
        # Получаем КБЖУ из вашего JSON
        nutrition = self.nutrition_db.get(label, "Nutrition data not found")

        probs = torch.nn.functional.softmax(logits, dim=-1)
        confidence = probs[0][predicted_class_idx].item()

        return {
            "dish": label,
            "confidence": round(confidence, 4),
            "nutrition_per_100g": nutrition
        }