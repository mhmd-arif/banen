# Use Python slim image to reduce size
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the project files to the container
COPY . /app

# Install dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data into /tmp/nltk_data
RUN python download_nltk_data.py

# Expose port 5000 to the host machine
EXPOSE 5000

# Run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
