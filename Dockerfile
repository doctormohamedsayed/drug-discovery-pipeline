# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for RDKit and OpenBabel
RUN apt-get update && apt-get install -y \
    libxrender1 \
    libxext6 \
    libsm6 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Expose port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Adjust the Flask port to match Hugging Face (7860) or general Docker (8080)
# We will use gunicorn to serve it
CMD ["gunicorn", "-b", "0.0.0.0:7860", "wsgi:app"]
