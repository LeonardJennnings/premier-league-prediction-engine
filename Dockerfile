# Start from a known Python (close to what we use locally).
FROM python:3.13-slim

# Work inside /app in the container.
WORKDIR /app

# Install deps first (better Docker layer caching).
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project (see .dockerignore for what is excluded).
COPY . .

# Streamlit port.
EXPOSE 8501

# Bind 0.0.0.0 so the app is reachable from outside the container.
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
