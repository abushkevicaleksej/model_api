FROM python:3.10-slim

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем всё содержимое папки app
COPY ./app ./app

# Предварительная загрузка весов модели из Hugging Face в образ
RUN python -c "from transformers import AutoModelForImageClassification, AutoImageProcessor; \
    AutoModelForImageClassification.from_pretrained('BinhQuocNguyen/food-recognition-model'); \
    AutoImageProcessor.from_pretrained('BinhQuocNguyen/food-recognition-model')"

# Открываем порт
EXPOSE 8000

# Запуск
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]