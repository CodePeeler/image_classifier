# Image Classifier

### Key Features

* Automatic label generation for training and validation datasets.


* Model Management features to manage lifecycle.


* Training parameters for tweaking learning algorithm.


* Visualization of Neural Network.


* User management

___

## Docker

### Build and tag a Docker image
```bash
docker build -t <your_tag_name>/image_classifier .
```

### Start Docker container
```bash
docker run -p 8000:8000 --name <your_container_name> <your_tag_name>/image_classifier
```
