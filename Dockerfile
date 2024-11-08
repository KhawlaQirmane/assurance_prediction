# Use the official Python image as a base image
FROM python:3.8-slim

# Install system dependencies for numerical packages
RUN apt-get update && \
    apt-get install -y build-essential gfortran libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements2.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements2.txt && \
    pip install --no-cache-dir --upgrade numpy

# Copy the rest of the application's code
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Set Flask environment variable
ENV FLASK_APP=api.py

# Command to run the application
CMD ["flask", "run", "--host=0.0.0.0"]
