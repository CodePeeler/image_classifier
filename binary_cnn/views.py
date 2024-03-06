import os
import json
import zipfile

import keras
from django.http import JsonResponse
from django.shortcuts import render, redirect

import binary_cnn.ml_cnn as ml
from binary_cnn.forms import DatasetForm, BinaryModelForm, TrainConfigForm, ImageForm
from binary_cnn.models import DSType, DSCategory, Dataset, Status, BinaryModel, Image, TrainConfig
import binary_cnn.utilities as util


def home(request):
    return render(request, 'binary_cnn/home.html')


def start(request):
    return render(request, 'binary_cnn/get_started.html')


def datasets(request):
    # Get all datasets.
    data_sets = Dataset.objects.order_by('-date_added')

    context = {'datasets': data_sets}
    return render(request, 'binary_cnn/datasets.html', context)


def add_dataset(request):
    if request.method != 'POST':
        # No data submitted, create a blank form.
        form = DatasetForm()
    else:
        # POST data submitted; process data.
        form = DatasetForm(data=request.POST, files=request.FILES)
        if form.is_valid():
            new_dataset = form.save(commit=False)
            try:
                # Set dataset's name equal to file name - minus ext.
                ds_name = new_dataset.save_dir.name.split('.')[0]
                new_dataset.name = ds_name
                new_dataset.save()

                # Extract zip from the default tmp folder to dataset root dir.
                zip_ref = zipfile.ZipFile(new_dataset.save_dir, 'r')
                ds_root = new_dataset.save_dir.name.split('tmp')[0]+new_dataset.name
                zip_ref.extractall(ds_root)
                zip_ref.close()

                # Update the dataset's save_dir to point to the dataset root dir.
                new_dataset.save_dir = ds_root
                new_dataset.save()

            except zipfile.BadZipFile:
                new_dataset.delete()
                raise ValueError("The file is not a valid zip file.")
            except Exception as e:
                print(f"Error adding dataset : {e}")
            finally:
                # clear tmp folder.
                util.rm_files('binary_cnn/datasets/tmp')

        return redirect('binary_cnn:datasets')

    context = {'form': form}
    return render(request, 'binary_cnn/add_dataset.html', context)


def delete_datasets(request):
    # Delete a batch of BinaryModel instances.
    raw_body = request.body
    body_str = raw_body.decode('utf-8')
    json_data = json.loads(body_str)
    ids = json_data.get('ids')

    for dataset_id in ids:
        try:
            dataset = Dataset.objects.get(id=dataset_id)
            dataset.delete()
        except Dataset.DoesNotExist:
            print("Dataset does not exist")

    # Doesn't refresh page!!!
    return redirect('binary_cnn:datasets')


def models(request):
    # Get all the trained models.
    binary_models = BinaryModel.objects.order_by('-date_added')

    context = {'binary_models': binary_models}
    return render(request, 'binary_cnn/models.html', context)


def model(request, model_id):
    try:
        # Get the requested model
        binary_model = BinaryModel.objects.get(id=model_id)
        train_config = None
        if binary_model.status == Status.TRAINED:
            train_config = TrainConfig.objects.get(binary_model=binary_model)
        context = {'binary_model': binary_model, 'train_config': train_config}
        return render(request, 'binary_cnn/model.html', context)
    except BinaryModel.DoesNotExist:
        # Handle the case where the object does not exist
        print("Object does not exist")
    return redirect('binary_cnn:models')


def delete_models(request):
    json_data_str = request.POST.get('json')
    json_data = json.loads(json_data_str)
    ids = json_data.get('ids')

    for model_id in ids:
        try:
            binary_model = BinaryModel.objects.get(id=model_id)
            binary_model.delete()
        except BinaryModel.DoesNotExist:
            print("BinaryModel does not exist")

    # Return json.
    data = {'msg': 'Models deleted!'}
    return JsonResponse(data)


# TODO Do you really need this method - could just use the above!
def delete_model(request, model_id):
    # Delete a single BinaryModel instances.
    try:
        binary_model = BinaryModel.objects.get(id=model_id)
        binary_model.delete()
    except BinaryModel.DoesNotExist:
        # Handle the case where the object does not exist
        print("Object does not exist")
    return redirect('binary_cnn:models')


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
    try:
        binary_model = BinaryModel.objects.get(id=model_id)

        if request.method != 'POST':
            train_form = TrainConfigForm()

        else:
            train_form = TrainConfigForm(data=request.POST)
            # Ensures all fields are fill out etc.
            if train_form.is_valid():

                # If retraining then delete old TrainConfig (BM to TC is 1 to 1)
                if binary_model.status == Status.TRAINED:
                    old_train_config = TrainConfig.objects.get(binary_model=model_id)
                    old_train_config.delete()
                    binary_model.status = Status.UNTRAINED
                    binary_model.save()

                # Create a new TrainingConfig instance.
                train_config = train_form.save(commit=False)

                # Load the model
                saved_model_path = binary_model.save_dir + '/' + binary_model.name
                ml_model = keras.models.load_model(saved_model_path)

                # Compile the model
                ml_model = ml.compile_model(ml_model)

                # Update the model's status
                binary_model.status = Status.TRAINING
                binary_model.save()

                # Train the model.
                training_dir = train_config.training_ds.save_dir.name
                validation_dir = train_config.validation_ds.save_dir.name
                ml_model = ml.train_model(ml_model, training_dir, validation_dir)

                # Save the ml model to the save dir of the BinaryModel.
                ml_model.save(saved_model_path)

                # Update the model's status
                binary_model.status = Status.TRAINED
                binary_model.save()

                # Associate TrainingConfig's binary_model field with the BinaryModel.
                train_config.binary_model = binary_model
                train_config.save()
                data = {'msg': "Your model has been trained"}
                return JsonResponse(data)

        context = {'train_form': train_form, 'binary_model': binary_model}
        return render(request, 'binary_cnn/train.html', context)

    except BinaryModel.DoesNotExist:
        # Handle the case where the object does not exist
        print("Object does not exist")
    return redirect('binary_cnn:models')


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
        train_config = TrainConfig.objects.get(binary_model=binary_model.id)

        # Get the datasets classes - used for classification of images.
        classes = train_config.training_ds.class_labels.split(',')

        #CLASS_DESC = ['horse', 'human']
        my_model = keras.models.load_model(bm_path)
        classification = ml.classify(my_model, formatted_img, classes)

        image.img_classification = classification
        image.classified = True
        image.save()

        img_file_name = img_path.split('/')[-1]
        context = {'classification': classification, 'img_file_name': img_file_name}

    return render(request, 'binary_cnn/classify_result.html', context)
