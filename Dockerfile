# ----------------------------------------
# 1. Base Python Image
# ----------------------------------------
FROM python:3.10-slim

# ----------------------------------------
# 2. Prevent OpenCV errors by installing system libs
# ----------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------
# 3. Create working directory
# ----------------------------------------
WORKDIR /app

# ----------------------------------------
# 4. Copy dependency file
# ----------------------------------------
COPY requirements.txt .

# ----------------------------------------
# 5. Install Python dependencies
# ----------------------------------------
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------
# 6. Copy the entire project
# ----------------------------------------
COPY . .

# ----------------------------------------
# 7. Expose port
# ----------------------------------------
EXPOSE 5000

# ----------------------------------------
# 8. Command to run Flask
# ----------------------------------------
CMD ["python", "app.py"]
