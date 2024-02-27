function hideContent(event) {
    document.getElementById("message").style.visibility = "hidden";
    document.getElementById("classify_form").style.visibility = "hidden";
    event.preventDefault();
}

function updatePage(img_id, model_id) {
    var xhr = new XMLHttpRequest();

    alert('/summary_update/'+img_id+'/'+model_id)
    xhr.open('GET', 'summary_update/'+img_id+'/'+model_id);
    xhr.onload = function() {
        if (xhr.status === 200) {
            var data = JSON.parse(xhr.responseText);
            displayData(data);
        }
        else {
            alert('Request failed. Returned status of ' + xhr.status);
        }
    };
    xhr.send();
}

function displayData(data) {
    document.getElementById("update").style.visibility = "visible";
    document.getElementById('result').innerHTML = data.classification;
}
