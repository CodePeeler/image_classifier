// Return the count for rows that have status of Training.
// Set Training text color to red.
// Set Untrained text color to orange.
// Set Trained text color to green.
function getNumOfRowsWithTrainingStatus(){
    var count = 0;
    var statusElements = document.getElementsByClassName("status")

    for (var i = 0; i < statusElements.length; i++) {
        var statusText = statusElements[i].textContent;
        if(statusText=='Training'){
            statusElements[i].style.color = "red";
            count++;
        }
        else if(statusText=='Untrained'){
            statusElements[i].style.color = "orange";
        }
        else if (statusText=='Trained'){
            statusElements[i].style.color = "green";
        }
    }
    return count
}

function refreshPage() {
    window.location.reload();
}


// Check if any model's status is in 'training' and if so keep refreshing
// page until the status changes.
function checkTextValueOnLoad() {
    // Initialise.
    var numOfRowsWithTrainingStatus = getNumOfRowsWithTrainingStatus()

    if(numOfRowsWithTrainingStatus > 0) {
        setTimeout(refreshPage, 2000);
    }
}

checkTextValueOnLoad();


function toggleAll(source) {
    var checkboxes = document.querySelectorAll('input[type="checkbox"]');
    for (var i = 0; i < checkboxes.length; i++) {
        checkboxes[i].checked = source.checked;
    }
}


// Fetch API to update page on submit of form.
document.getElementById('myForm').addEventListener('submit', function(event) {

    event.preventDefault(); // Prevent default form submission

    // Get all checkboxes
    var checkboxes = document.querySelectorAll('.rowCheckbox');
    // Array to store IDs of selected rows to delete
    var idsToDelete = [];

    // Loop through checkboxes to find selected ones
    checkboxes.forEach(function(checkbox) {
        if (checkbox.checked) {
            // If checkbox is checked, add its ID to idsToDelete array
            idsToDelete.push(checkbox.getAttribute('data-id'));
        }
    });

    // Make API call to delete models from server
    if (idsToDelete.length > 0) {
        var data = { ids: idsToDelete };

        // Get form data
        const formData = new FormData(this);

        formData.append('json', JSON.stringify(data));

        // Send form data to the server
        fetch('delete/', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            console.log(data)
            // Delete selected rows from table if API call is successful
            idsToDelete.forEach(function(id) {
                document.querySelector('[data-id="' + id + '"]').closest('tr').remove();
            });
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
