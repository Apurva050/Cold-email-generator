# Use slim Python base image
FROM python:3.10-slim

# Set working directory inside container
WORKDIR /app

# Copy everything to container
COPY . .

# Install pip dependencies
RUN pip install --upgrade pip 
RUN pip install -r requirements.txt

# Streamlit runs on port 8501
EXPOSE 8501

# Set environment variable so Streamlit runs without asking for email
ENV STREAMLIT_DISABLE_WELCOME_MESSAGE=true

# Run Streamlit app from app/main.py
CMD ["streamlit", "run", "app/main.py", "--server.port=8501", "--server.address=0.0.0.0"]
