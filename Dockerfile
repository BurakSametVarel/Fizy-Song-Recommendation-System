# Temel Python imajı (librosa ve bilimsel paketler için önerilen imaj)
FROM python:3.10-slim

# Gerekli sistem paketleri (librosa ve ses işleme için)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli dosyaları kopyala
COPY requirements.txt .
COPY . .

# Gereken Python paketlerini yükle
RUN pip install --no-cache-dir -r requirements.txt

# Flask uygulaması 5000 portunu dinler
EXPOSE 5000

# Uygulamayı başlat
CMD ["python", "app.py"]
