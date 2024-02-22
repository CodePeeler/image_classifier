
// JavaScript to perform action based on radio button selection
document.querySelectorAll('input[type="radio"]').forEach(function(radio) {
    radio.addEventListener('change', function() {
        var selectedOption = document.querySelector('input[name="options"]:checked');
        // Check if any radio button is checked
        if (selectedOption.value == 'create') {
            document.getElementById("create_form").style.visibility = "visible";
        } else if(selectedOption.value == 'retrain') {
            document.getElementById("create_form").style.visibility = "hidden";
        } else {
            resultElement.textContent = "No option selected";
        }
    });
});

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
        displayFileContent(data.model_summary_txt)
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
        .then(data => {
            document.getElementById("create").checked = false;
            document.getElementById('create_form').reset();
            document.getElementById("create_form").style.visibility = "hidden";

            // Update the content of the 'fileContent' element with the text file content
            document.getElementById('fileContent').textContent = data;
        })
        .catch(error => {
            console.error('Error fetching file:', error);
        });
}
