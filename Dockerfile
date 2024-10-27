# Use Python slim image to keep it lightweight
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all files from your project into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Expose port 5000 to the host
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
