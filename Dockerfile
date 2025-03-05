# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /agent

# Copy the current directory contents into the container at /agent
COPY . /agent

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 80 available to the world outside this container
EXPOSE 80

# Copy .env file to the container
COPY .env .env

# Export environment variables from .env file
RUN export $(grep -v '^#' .env | xargs)

# Run pyinoke task when the container launches
CMD ["python","-m", "invoke"]