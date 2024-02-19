from django.shortcuts import render

from binary_cnn.forms import ImageForm


def home(request):
    return render(request, 'binary_cnn/home.html')


def upload(request):
    if request.method != 'POST':
        # No data submitted, create a blank form.
        form = ImageForm()
    else:
        # POST data submitted; process data.
        form = ImageForm(data=request.POST, files=request.FILES)
        # Ensures all fields are fill out etc.
        if form.is_valid():
            form.save()
            return render(request, 'binary_cnn/success.html')
    # Display a blank or invalid form.
    context = {'form': form}
    return render(request, 'binary_cnn/upload.html', context)
