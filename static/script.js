document.addEventListener("DOMContentLoaded", () => {
    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message");
    const chatArea = document.getElementById("chat-area");
    const centeredContent = document.getElementById("centered-content");

    chatForm.addEventListener("submit", (event) => {
        event.preventDefault(); // Prevent form from submitting the default way
        const description = messageInput.value;
        console.log("Description:", description);  // Add this line for debugging

        if (description.trim() === "") {
            alert("Please enter a description.");
            return;
        }

        addMessageToChat(description, true);
        messageInput.value = "";  // Clear the input field

        // Hide logo and description
        centeredContent.style.display = "none";

        // Send the message to the backend as JSON
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ description: description })
        })
        .then(response => response.json())
        .then(data => {
            console.log("Backend response:", data);  // Debug log
            if (data.error) {
                addMessageToChat("There was an error processing your request.", false);
            } else {
                const backendResponse = `Risks: ${data.risks.join(', ')}\n\nExplanation:\n${data.explanation}\n\nMitigation Plan:\n${data.mitigation_plan}`;
                addMessageToChat(backendResponse, false);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            addMessageToChat("There was an error connecting to the backend.", false);
        });
    });

    function addMessageToChat(message, isUser) {
        const chatDiv = document.createElement("div");
        chatDiv.classList.add("chat-message");

        const messageSpan = document.createElement("span");
        messageSpan.classList.add("message");
        messageSpan.textContent = message;

        if (isUser) {
            chatDiv.classList.add("user-message");
        } else {
            chatDiv.classList.add("backend-message");
        }

        chatDiv.appendChild(messageSpan);
        chatArea.appendChild(chatDiv);
        chatArea.scrollTop = chatArea.scrollHeight; // Scroll to the bottom
    }
});
