FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY . .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu \
    torch==2.5.1+cpu torchvision==0.20.1+cpu torchaudio==2.5.1+cpu \
    gunicorn soundfile && \
    pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "gunicorn app:app --bind 0.0.0.0:${PORT:-5030} --workers 1 --threads 2 --timeout 120"]
