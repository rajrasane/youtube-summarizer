const btn = document.getElementById("summarizeBtn");
const output = document.getElementById("summaryOutput");
const loader = document.getElementById("loader");

btn.addEventListener("click", async () => {
    const url = document.getElementById("youtubeUrl").value.trim();

    if (!url) {
        alert("Please enter a YouTube URL");
        return;
    }

    output.textContent = "";
    loader.classList.remove("hidden");
    btn.disabled = true;

    try {
        const res = await fetch("http://127.0.0.1:5000/summarize", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });

        const data = await res.json();

        if (data.success) {
            output.textContent = data.summary;
        } else {
            output.textContent = "Error: " + data.error;
        }

    } catch (err) {
        output.textContent = "Server error";
    } finally {
        loader.classList.add("hidden");
        btn.disabled = false;
    }
});
