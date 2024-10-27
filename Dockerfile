# Use Python slim image to keep it lightweight
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy all project files into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the necessary NLTK data
RUN python -m nltk.downloader -d /usr/local/nltk_data punkt stopwords punkt_tab

# Set environment variable for NLTK data path
ENV NLTK_DATA=/usr/local/nltk_data

# Expose port 5000 to the host
EXPOSE 5000

# Command to run the Flask app using Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
