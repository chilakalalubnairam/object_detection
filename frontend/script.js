
async function detectObject() {
    const response = await fetch("http://127.0.0.1:5000/detect");
    const data = await response.json();

    document.getElementById("outputImage").src =
        "http://127.0.0.1:5000/image/" + data.image + "?t=" + new Date().getTime();

    document.getElementById("labels").innerText =
        "Detected: " + data.labels.join(", ");
}