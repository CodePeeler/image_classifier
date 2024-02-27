from concurrent.futures import ThreadPoolExecutor
import os
import stat

from django.http import JsonResponse
from django.shortcuts import render, redirect

from binary_cnn.forms import BinaryModelForm, TrainingConfigForm, ImageForm
from binary_cnn.models import Status, BinaryModel, Image, TrainingConfig
import keras

import binary_cnn.ml_cnn as ml


def home(request):
    return render(request, 'binary_cnn/home.html')


# This should only be called when the current status of a model is
# actively training - will be called with javascript using a callback
# as the check_status is blocking.
def update_status(request):
    # TODO, Think how you would get an update for a given model.
    #  - currently we assume only one model can be in a training state.
    data = {'status': ml.check_status()}
    return JsonResponse(data)


def models(request):
    # Get all the trained models.
    binary_models = BinaryModel.objects.all()

    context = {'binary_models': binary_models}
    return render(request, 'binary_cnn/models.html', context)


# Javascript will call this method.
def create(request):
    if request.method != 'POST':
        create_form = BinaryModelForm()
        context = {'create_form': create_form}
        return render(request, 'binary_cnn/create.html', context)
    else:
        create_form = BinaryModelForm(data=request.POST)

        # Ensures all fields are fill out etc.
        if create_form.is_valid():
            # Create a new instance of BinaryModel.
            new_bm = create_form.save(commit=False)

            # Create a machine learning model (i.e. tensorflow)
            ml_model = ml.create_model()

            # Save the ml model to 'save_dir' of the BinaryModel instances.
            path = os.path.join(new_bm.save_dir, new_bm.name)
            ml_model.save(path)

            # Create static and absolute paths for the model's summary text file.
            model_summary_name = new_bm.name + '_summary.txt'
            model_summary_static_path = os.path.join('/static/text/', model_summary_name)
            model_summary_abs_path = os.getcwd() + '/binary_cnn' + model_summary_static_path

            # Redirect print output to the file.
            with open(model_summary_abs_path, 'w') as f:
                ml_model.summary(print_fn=lambda x: f.write(x + '\n'))

            # Add the absolute path to the BinaryModel instances - required for clean-up!
            new_bm.path_model_summary = model_summary_abs_path

            # Commit changes to the db.
            new_bm.save()

            # Return json.
            msg = f'Success: {new_bm.name} created!'
            data = {'model_id': new_bm.id, 'model_summary_txt': model_summary_static_path, 'msg': msg}
            return JsonResponse(data)


def train(request, model_id):
    # TODO code to check if model has already been trained
    binary_model = BinaryModel.objects.get(id=model_id)

    if request.method != 'POST':
        train_form = TrainingConfigForm()

    else:
        train_form = TrainingConfigForm(data=request.POST)
        # Ensures all fields are fill out etc.
        if train_form.is_valid():
            # Create a new TrainingConfig instance.
            new_tc = train_form.save(commit=False)

            # Load the model
            SAVE_MODEL_DIR = binary_model.save_dir
            SAVED_MODEL_NAME = binary_model.name

            ml_model = keras.models.load_model(SAVE_MODEL_DIR + '/' + SAVED_MODEL_NAME)
            ml_model = ml.compile_model(ml_model)
            TRAINING_DIR = "/Users/simondornan/Documents/2023/ML_and_CS/01_ML_Path/06_My_ML_Projects/image_classifier/binary_cnn/data/horse-or-human"
            VALIDATION_DIR = "/Users/simondornan/Documents/2023/ML_and_CS/01_ML_Path/06_My_ML_Projects/image_classifier/binary_cnn/data/validation-horse-or-human"

            # Update the model's status
            binary_model.status = Status.TRAINING
            binary_model.save()
            ml_model = ml.train_model(ml_model, TRAINING_DIR, VALIDATION_DIR)

            # Save the ml model to the save dir of the BinaryModel.
            ml_model.save(SAVE_MODEL_DIR + "/" + SAVED_MODEL_NAME)

            # Update the model's status
            binary_model.status = Status.TRAINED
            binary_model.save()

            # Associate TrainingConfig's binary_model field with the BinaryModel.
            new_tc.binary_model = binary_model
            new_tc.save()

            """ ------------------------- parallel thread ----------------------------------------"""
            # Start the long-running task in a separate thread
            #with ThreadPoolExecutor() as executor:
                #executor.submit(long_running_task_test)

            #with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                # Submit the task to the executor asynchronously
                # future = executor.submit(train_task(binary_model, new_tc))
                #future = executor.submit(long_running_task_test)

                # Add the callback function to be executed when the result is available
                #future.add_done_callback(callback_fn)

                # This line of code will be executed immediately after submitting the task
            #print("Task submitted, continuing with other work...")
            #return redirect('binary_cnn:models')
            data = {'msg': "Your model has been trained"}
            return JsonResponse(data)

    context = {'train_form': train_form, 'binary_model': binary_model}
    return render(request, 'binary_cnn/train.html', context)


def classify_form(request):
    # Get all the trained models.
    trained_binary_models = BinaryModel.objects.filter(status=Status.TRAINED)

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
