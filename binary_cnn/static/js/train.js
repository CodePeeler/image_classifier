document.getElementById('train_form').addEventListener('submit', function(event) {
    event.preventDefault(); // Prevent default form submission

    // Get form data
    const formData = new FormData(this)
    document.getElementById('submit').style.visibility = "hidden";
    document.getElementById('back_to_models_button').style.visibility = "visible";
    document.getElementById('text_box').style.visibility = "visible";

    // Get all elements with the specified CSS class
    var elements = document.getElementsByClassName("loader");

    // Iterate over the elements and set their visibility property to "hidden"
    for (var i = 0; i < elements.length; i++) {
        elements[i].style.visibility = "visible";
    }

    // Send form data to the server
    fetch('/train/'+model_id.value, {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        document.getElementById('text_box').style.visibility = "hidden";

        for (var i = 0; i < elements.length; i++) {
            elements[i].style.visibility = "hidden";
        }

        document.getElementById('message').textContent = data.msg;

    })
    .catch(error => {

        document.getElementById('text_box').style.visibility = "hidden";

        for (var i = 0; i < elements.length; i++) {
            elements[i].style.visibility = "hidden";
        }
        document.getElementById('message').textContent = Error;
        console.error('Error:', error);
    });
});