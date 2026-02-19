function toggleCode(taskId, btnElement) {
    const answerBlock = document.getElementById(taskId);
    
    // Check computed style for robustness
    const isHidden = window.getComputedStyle(answerBlock).display === 'none';

    if (isHidden) {
        answerBlock.style.display = 'block';
        btnElement.innerText = "Hide Answer";
        btnElement.classList.add('hide-mode');
        
        // Simple scroll into view for better UX on mobile
        answerBlock.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        
    } else {
        answerBlock.style.display = 'none';
        btnElement.innerText = "Show Answer";
        btnElement.classList.remove('hide-mode');
    }
}

// Function to copy code to clipboard
function copyToClipboard(button) {
    // Find the code block closest to the button
    const container = button.closest('.answer-container');
    const codeBlock = container.querySelector('code');
    
    // Get text content
    const codeText = codeBlock.innerText;
    
    // Use the clipboard API
    navigator.clipboard.writeText(codeText).then(() => {
        // Visual feedback
        const originalText = button.innerText;
        button.innerText = "Copied!";
        button.style.color = "#10b981"; // Success Green
        
        setTimeout(() => {
            button.innerText = originalText;
            button.style.color = ""; // Reset
        }, 2000);
    }).catch(err => {
        console.error('Failed to copy text: ', err);
    });
}
