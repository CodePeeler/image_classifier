import os
import stat

from django.http import JsonResponse
from django.shortcuts import render, redirect

from binary_cnn.forms import BinaryModelForm, ImageForm
from binary_cnn.models import Image
import keras

import binary_cnn.ml_cnn as ml


def home(request):
    return render(request, 'binary_cnn/home.html')


def ml_models(request):
    if request.method != 'POST':
        form = BinaryModelForm()
    else:
        form = BinaryModelForm(data=request.POST)
        # Ensures all fields are fill out etc.
        if form.is_valid():
            # Create a new instance of BinaryModel.
            new_bm = form.save(commit=False)

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
            data = {'model_summary_txt': model_summary_static_path}
            return JsonResponse(data)

    context = {'form': form}
    return render(request, 'binary_cnn/ml_models.html', context)


def upload(request):
    if request.method != 'POST':
        # No data submitted, create a blank form.
        form = ImageForm()
    else:
        # POST data submitted; process data.
        form = ImageForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            new_image = form.save()
            return redirect('binary_cnn:classify_summary', img_id=new_image.id)

    # Display a blank or invalid form.
    context = {'form': form}
    return render(request, 'binary_cnn/upload.html', context)


def classify_summary(request, img_id):
    image = Image.objects.get(id=img_id)
    img_path = image.img_file.path
    file_name = img_path.split('/')[-1]
    context = {'img_id': img_id, 'file_name': file_name}
    return render(request, 'binary_cnn/classify_summary.html', context)


#Calls Machine Learning classify function.
def summary_update(request, img_id):
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
