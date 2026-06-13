FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt

# 3. CRITICAL STEP: Copy your actual code (main.py, ImageProcessor.py, etc.) into the container
COPY . .

# 4. Start the server automatically at runtime (Notice it uses CMD, not RUN)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]