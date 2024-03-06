function toggleAll(source) {
    var checkboxes = document.querySelectorAll('input[type="checkbox"]');
    for (var i = 0; i < checkboxes.length; i++) {
        checkboxes[i].checked = source.checked;
    }
}


function deleteDatasets() {
    alert("Deleting datasets!")
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
