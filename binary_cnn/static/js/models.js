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