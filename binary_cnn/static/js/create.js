// Fetch API to update page on submit of form.
document.getElementById('create_form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get form data
    const formData = new FormData(this);

    // Send form data to the server
    fetch('', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        // Get a reference to the element you want to remove
        var elementCreateForm = document.getElementById('create_form');

        // Get a reference to the parent node of the element
        var parentElement = elementCreateForm.parentNode;

        // Remove the element from its parent node
        parentElement.removeChild(elementCreateForm);

        document.getElementById('back_to_models_button').style.visibility = "visible";
        document.getElementById('train_button').style.visibility = "visible";

        // Update model_created_msg div.
        var createdMsgElement = document.getElementById("model_created_msg");
        createdMsgElement.textContent = data.msg;

        // Display summary title
        var archSummaryElement = document.getElementById("arch_summary");
        archSummaryElement.textContent = "Model Architecture Summary";

        displayFileContent(data.model_summary_txt)

        // Set the model_id input field for the train button.
        setHiddenValue(data.model_id)
    })
    .catch(error => {
        console.error('Error:', error);
    });
});

function displayFileContent(path) {
    // Path to the text file
    var filePath = path;

    // Fetch the text file, reset and hide form, and unselect radio button.
    fetch(filePath)
        .then(response => response.text())
        .then(file_data => {
            // Update the content of the 'fileContent' element with the text file content
            document.getElementById('fileContent').textContent = file_data;
        })
        .catch(error => {
            console.error('Error fetching file:', error);
        });
}

// Set the model id on hidden input field - associated with the train button.
function setHiddenValue(model_id) {
    // Get the hidden input element by its ID
    var hiddenInput = document.getElementById("hidden_model_id");

    // Set the value of the hidden input field
    hiddenInput.value = model_id;
}

function trainRequest() {
    // Define the URL
    var base_url = window.location.origin;
    var train_rel_url = '/train/'
    var model_id = hidden_model_id.value;

    var train_abs_url = base_url + train_rel_url + model_id

    // Make the URL request
    window.location.href = train_abs_url;
}
