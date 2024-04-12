# Use a base image with Python 3.9.
FROM python:3.9

# Set working directory in the container.
WORKDIR /opt/image_classifier

# Install system dependencies for HDF5
RUN apt-get update && apt-get install -y \
    libhdf5-dev \
    && rm -rf /var/lib/apt/lists/*

# Create directories for the Django project
RUN mkdir binary_cnn
RUN mkdir image_classifier

# Copy application code
COPY ./binary_cnn ./binary_cnn
COPY ./image_classifier ./image_classifier
COPY manage.py .

# Install Python dependencies
RUN pip install --no-cache-dir \
    Django \
    h5py \
    keras \
    matplotlib \
    numpy \
    pillow \
    wget \
    scikit-learn \
    scipy \
    tensorflow \
    zipp

# Expose the port for the development server
EXPOSE 8000

# Run the Django development server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]