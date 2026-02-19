function toggleCode(taskId, btnElement) {
    const answerBlock = document.getElementById(taskId);
    
    if (answerBlock.style.display === "none" || answerBlock.style.display === "") {
        answerBlock.style.display = "block"; // Show both code and explanation
        btnElement.innerText = "Hide Answer";
        btnElement.style.backgroundColor = "#e74c3c"; // Red for hide
    } else {
        answerBlock.style.display = "none";
        btnElement.innerText = "Show Answer";
        btnElement.style.backgroundColor = "#27ae60"; // Green for show
    }
}
