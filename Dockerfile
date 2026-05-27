# FastAPI inference server containerization
# Note: Ensure CPU PyTorch is installed to avoid downloading the ~2.5GB CUDA build, since this is for inference only.

# Base image: Use slim python for a smaller footprint
FROM python:3.10-slim

# Working dir: Set the directory where the app will live
WORKDIR /app

# Copy requirements file: Copy the file to the container
COPY requirements.txt .

# Install dependencies: The CPU PyTorch wheel URL must be used to avoid fetching the CUDA build
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy app directory and models directory
COPY app/ ./app/
COPY models/ ./models/

# Expose port: 8000 for FastAPI
EXPOSE 8000

# Entrypoint: Start the Uvicorn server to host the FastAPI app
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
