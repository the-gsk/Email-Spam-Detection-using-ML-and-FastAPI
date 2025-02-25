# Use official Python image
FROM python:3.11

# Set working directory in the container
WORKDIR /app

# Copy the application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install -r requirements.txt

# Expose port for FastAPI
EXPOSE 8000

# Run the application
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
