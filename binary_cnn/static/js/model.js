document.getElementById('modelForm').addEventListener('submit', function(event) {
    event.preventDefault();

    var id = document.getElementById('model_id').value
    var idsToDelete = [id];
    var data = { ids: idsToDelete };

    const formData = new FormData(this);

    formData.append('json', JSON.stringify(data));

    fetch('/models/delete/', {
        method: 'POST',
        body: formData
    })
    .then(response => {
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        return response.json();
    })
    .then(data => {
        console.log('Response from server:', data);
        // Redirect
        window.location.href = '/models/';
    })
    .catch(error => {
        console.error('Error:', error);
    });
});
