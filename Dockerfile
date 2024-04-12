# Use a base image with Python 3.9.
FROM python:3.9

# Set working directory in the container.
WORKDIR /opt/image_classifier

RUN mkdir binary_cnn
RUN mkdir image_classifier

# copy application code
COPY ./binary_cnn ./binary_cnn
COPY ./data ./data
COPY manage.py .

# Install dependencies.
RUN pip install --no-cache-dir \
    Django \
    keras \
    matplotlib \
    numpy \
    pillow \
    wget \
    scikit-learn \
    scipy \
    tensorflow \
    zipp

# Expose the port for dev server.
EXPOSE 8000

# Run the application.
CMD ["python", "manage.py runserver", "0.0.0.0:8000"]
