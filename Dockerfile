# Use PyTorch with CPU support for deployment if no GPU needed
FROM python:3.10-slim

# Install system dependencies for OpenCV and other packages
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Install uvicorn and fastapi specifically
RUN pip install --no-cache-dir uvicorn fastapi python-multipart

# Copy the rest of the application
COPY . .

# Expose the port GCP Cloud Run expects
EXPOSE 8080

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
