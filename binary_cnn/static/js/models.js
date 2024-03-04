// Function to check the text value on load
function checkTextValueOnLoad() {
    // Initialise.
    var numOfRowsWithTrainingStatus = getNumOfRowsWithTrainingStatus()

    if(numOfRowsWithTrainingStatus > 0) {
        setTimeout(refreshPage, 2000);
    }
}

// Gets the count and set all in training to have red text.
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
// page until the status changes. Note, not affective for multi tabs use;
// where training is trigger on one tab page then the other page i.e models
// page will not be updated automatically, i.e. the user must refresh page.
checkTextValueOnLoad();

function toggleAll(source) {
    var checkboxes = document.querySelectorAll('input[type="checkbox"]');
    for (var i = 0; i < checkboxes.length; i++) {
        checkboxes[i].checked = source.checked;
    }
}


function deleteModels() {
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

    // Make API call to delete objects from server
    if (idsToDelete.length > 0) {
        // Assuming there's an API endpoint "/delete_objects" to delete objects

        var data = { ids: idsToDelete };

        // Make a POST request to the API endpoint
        const csrfToken = document.querySelector('input[name="csrfmiddlewaretoken"]').value;
        fetch('delete/', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': csrfToken
            },
            body: JSON.stringify(data)
        })
        .then(response => {
            if (response.ok) {
                // Delete selected rows from table if API call is successful
                idsToDelete.forEach(function(id) {
                    document.querySelector('[data-id="' + id + '"]').closest('tr').remove();
                });
            } else {
                console.error('Error deleting objects:', response.statusText);
            }
        })
        .catch(error => {
            console.error('Error deleting objects:', error);
        });
    }
}
