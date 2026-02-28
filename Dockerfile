FROM python:3.10-slim

WORKDIR /app

# Install CPU-only PyTorch first (much smaller than CUDA version)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . .

# Expose port
EXPOSE 5000

# Run
CMD ["python", "messenger_bot.py"]
