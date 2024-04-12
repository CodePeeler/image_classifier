# Image Classifier

### Key Features

* Automatic label generation for training and validation datasets.
  

  __NB__ _When uploading your data (training/validation) for binary classification. Please structure your zip file as follows._
  
  Example cats-v-dogs.zip

    
      cats_or_dogs
        |
        +- cats
           |
           +- cat.0.jpg
              .
              .
              .
             cat.999.jpg
             
        +- dogs
           |
           +- dog.0.jpg
              .
              .
              .
            dog.999.jpg

* Model Management features to manage lifecycle.

      * Quickly create ML model using template
      * Save and retrain models              


* Training parameters for tweaking learning algorithm. 

      * Learning rate
      * Minimum accuracy
      * Number of epochs


* Performance Metrics:

      * Training v Validation accuracy/loss grpahs
      * Key Performace Indicators
          Precision
          Recall
          F1 Score
  
* User management

  _TODO_

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
