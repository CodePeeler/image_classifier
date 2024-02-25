import os
import stat

from django.http import JsonResponse
from django.shortcuts import render, redirect

from binary_cnn.forms import BinaryModelForm, TrainingConfigForm, ImageForm
from binary_cnn.models import BinaryModel, Image, TrainingConfig
import keras

import binary_cnn.ml_cnn as ml


def home(request):
    return render(request, 'binary_cnn/home.html')


def ml_models(request):

    if request.method != 'POST':
        bm_form_create = BinaryModelForm()
        bm_form_train = TrainingConfigForm()
        context = {'bm_form_create': bm_form_create, 'bm_form_train': bm_form_train}
        return render(request, 'binary_cnn/ml_models.html', context)
    else:
        if request.POST['process'] == 'create':
            bm_form_create = BinaryModelForm(data=request.POST)

            # Ensures all fields are fill out etc.
            if bm_form_create.is_valid():
                # Create a new instance of BinaryModel.
                new_bm = bm_form_create.save(commit=False)

                # Create a machine learning model (i.e. tensorflow)
                ml_model = ml.create_model()

                # Save the ml model to 'save_dir' of the BinaryModel instances.
                path = os.path.join(new_bm.save_dir, new_bm.name)
                ml_model.save(path)

                # Create static and absolute paths for the model's summary text file.
                model_summary_name = new_bm.name+'_summary.txt'
                model_summary_static_path = os.path.join('/static/text/', model_summary_name)
                model_summary_abs_path = os.getcwd()+'/binary_cnn'+model_summary_static_path

                # Redirect print output to the file.
                with open(model_summary_abs_path, 'w') as f:
                    ml_model.summary(print_fn=lambda x: f.write(x + '\n'))

                # Add the absolute path to the BinaryModel instances - required for clean-up!
                new_bm.path_model_summary = model_summary_abs_path

                # Commit changes to the db.
                new_bm.save()

                # Return the static path as JSON response.
                data = {'model_summary_txt': model_summary_static_path, 'bm_id': new_bm.id}
                return JsonResponse(data)

            context = {'bm_form_create': bm_form_create}
            return render(request, 'binary_cnn/ml_models.html', context)

        if request.POST['process'] == 'train':
            bm_form_train = TrainingConfigForm(data=request.POST)

            # Ensures all fields are fill out etc.
            if bm_form_train.is_valid():
                # Create a new TrainingConfig instance.
                new_tc = bm_form_train.save(commit=False)

                bm_id = request.POST["binary_model_id"]
                binary_model = BinaryModel.objects.get(id=bm_id)

                # Use the parameters from TrainingConfig to train your model.
                # Load the model
                SAVE_MODEL_DIR = binary_model.save_dir
                SAVED_MODEL_NAME = binary_model.name
                ml_model = keras.models.load_model(SAVE_MODEL_DIR +'/'+ SAVED_MODEL_NAME)

                ml_model = ml.compile_model(ml_model)
                TRAINING_DIR = "/Users/simondornan/Documents/2023/ML_and_CS/01_ML_Path/06_My_ML_Projects/image_classifier/binary_cnn/data/horse-or-human"
                VALIDATION_DIR = "/Users/simondornan/Documents/2023/ML_and_CS/01_ML_Path/06_My_ML_Projects/image_classifier/binary_cnn/data/validation-horse-or-human"

                ml_model = ml.train_model(ml_model, TRAINING_DIR, VALIDATION_DIR)

                # Save the ml model to the save dir of the BinaryModel.
                ml_model.save(SAVE_MODEL_DIR +"/"+ SAVED_MODEL_NAME)

                # Set trained to True and save to db.
                binary_model.is_trained = True
                binary_model.save()

                # Associate TrainingConfig to a BinaryModel and save to db.
                new_tc.binary_model = binary_model
                new_tc.save()

                data = {'msg': "Your model has been trained"}
                return JsonResponse(data)

            context = {'bm_form_train': bm_form_train}
            return render(request, 'binary_cnn/ml_models.html', context)


# --------------
def classify_form(request):
    # Get all the trained models.
    trained_binary_models = BinaryModel.objects.filter(is_trained=True)

    if request.method != 'POST':
        # No data submitted, create a blank form.
        form = ImageForm()
    else:
        # POST data submitted; process data.
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            new_image = form.save(commit=False)
            new_image.model_id = request.POST["model_id"]
            new_image = form.save()
            return redirect('binary_cnn:classify_result.html', image_id=new_image.id)

    # Display a blank or invalid form.
    context = {'form': form, 'trained_models': trained_binary_models}
    return render(request, 'binary_cnn/classify_form.html', context)


def classify_result(request, image_id):
    image = Image.objects.get(id=image_id)
    binary_model = BinaryModel.objects.get(id=image.model_id)

    # Check if images is already classified e.g. browser refresh!
    if image.is_classified:
        context = {'classification': image.img_classification}
    else:
        img_path = image.img_file.path

        # Normalize the image
        formatted_img = ml.format_img(img_path)

        # Load a ml model.
        bm_path = os.path.join(binary_model.save_dir, binary_model.name)

        # Get binary model's training config.
        training_config = TrainingConfig.objects.filter(binary_model=binary_model)[0]

        # Get the datasets classes - used for classification of images.
        classes = training_config.classification_classes.split(',')

        #CLASS_DESC = ['horse', 'human']
        my_model = keras.models.load_model(bm_path)
        classification = ml.classify(my_model, formatted_img, classes)

        image.img_classification = classification
        image.classified = True
        image.save()

        img_file_name = img_path.split('/')[-1]
        context = {'classification': classification, 'img_file_name': img_file_name}

    return render(request, 'binary_cnn/classify_result.html', context)


# --------------

def upload(request):
    # Show all trained models.
    trained_binary_models = BinaryModel.objects.filter(is_trained=True)

    if request.method != 'POST':
        # No data submitted, create a blank form.
        form = ImageForm()

    else:
        # POST data submitted; process data.
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            model_id = request.POST["model_id"]
            new_image = form.save()
            return redirect('binary_cnn:classify_summary', img_id=new_image.id, model_id=model_id)

    # Display a blank or invalid form.
    context = {'form': form, 'trained_models': trained_binary_models}
    return render(request, 'binary_cnn/upload.html', context)


def classify_summary(request, img_id, model_id):
    image = Image.objects.get(id=img_id)
    img_path = image.img_file.path
    file_name = img_path.split('/')[-1]
    context = {'img_id': img_id, 'model_id': model_id, 'file_name': file_name}
    return render(request, 'binary_cnn/classify_summary.html', context)


#Calls Machine Learning classify function.
def summary_update(request, img_id, model_id):
    image = Image.objects.get(id=img_id)

    # Check if images is classified e.g. due to browser refresh.
    if image.classified:
        data = {'classification': image.img_classification}
    else:
        img_path = image.img_file.path

        # Normalize the image
        formatted_img = ml.format_img(img_path)

        # Load a model.
        SAVE_MODEL_DIR = 'binary_cnn/saved_models/'
        SAVED_MODEL_NAME = 'my_model_v1'
        CLASS_DESC = ['horse', 'human']

        my_model = keras.models.load_model(SAVE_MODEL_DIR + SAVED_MODEL_NAME)
        class_result = ml.classify(my_model, formatted_img, CLASS_DESC)

        image.img_classification = class_result
        image.classified = True
        image.save()
        data = {'classification': class_result}
    return JsonResponse(data)
