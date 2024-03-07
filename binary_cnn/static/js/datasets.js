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

    // Make API call to delete datasets from server
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
             if (data.error_type == "ProtectedError") {
                document.getElementById('error_msg').textContent = data.error_msg
                document.getElementById('error_msg').style.color = "red";
            } else {
                // Delete selected rows from table if API call is successful
                idsToDelete.forEach(function(id) {
                    document.querySelector('[data-id="' + id + '"]').closest('tr').remove();
                });
            }
        })
        .catch(error => {
            console.error('Error:', error);
        });
    }
});
