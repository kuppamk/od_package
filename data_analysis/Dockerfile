FROM python:3.11-slim

# Update and install essential packages including libGL for OpenCV
RUN apt-get update && apt-get install -y \ 
    build-essential \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only labels (like json files)
COPY *.json ./

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
