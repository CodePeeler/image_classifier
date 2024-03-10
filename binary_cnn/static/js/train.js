function goBackToModelsPage() {
     window.location.href = '/models/';
}

document.getElementById('train_form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get form data
    const formData = new FormData(this)

    // Update UI; remove submit btn and add back to models btn.
    document.getElementById('submit').remove();

    // Create a new button element
    var bckToModelButton = document.createElement('button');

    // Set attributes for the new button.
    bckToModelButton.id = 'bck-to-models-btn2';
    bckToModelButton.classList.add("btn", "btn-sm", "btn-primary");
    bckToModelButton.type = 'button';
    bckToModelButton.textContent = 'Back to Models';
    bckToModelButton.onclick = goBackToModelsPage;

    // Get a reference to the form
    var form = document.getElementById('train_form');

    // Append the new button to the form
    form.appendChild(bckToModelButton);

    // Get all elements with the specified CSS class - used to show the spinner.
    var elements = document.getElementsByClassName("loader");
    // Iterate over the elements setting visibility to visible.
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.visibility = "visible";
    }

    // Show the user the status of the ml-model.
    document.getElementById('text_box').textContent = "Training...";

    // Send form data to the server
    fetch('/train/'+model_id.value, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        // Show the user the updated status of the ml-model.
        document.getElementById('text_box').textContent = data.msg;

        for (var i = 0; i < elements.length; i++) {
            elements[i].style.visibility = "hidden";
        }
    })
    .catch(error => {
        document.getElementById('text_box').style.visibility = "hidden";

        for (var i = 0; i < elements.length; i++) {
            elements[i].style.visibility = "hidden";
        }
        document.getElementById('text_box').textContent = Error;
        console.error('Error:', error);
    });
});
