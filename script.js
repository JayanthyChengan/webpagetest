let pipelineInstance = null;

// Load model ONCE
async function loadModel() {
    if (!pipelineInstance) {
        pipelineInstance = await window.transformers.pipeline(
            "feature-extraction",
            "Xenova/all-MiniLM-L6-v2"
        );
    }
    return pipelineInstance;
}

// Convert array → vector
function meanPool(tensor) {
    const arr = tensor.data;
    let result = 0;
    for (let i = 0; i < arr.length; i++) result += arr[i];
    return result / arr.length;
}

// Cosine similarity
function cosine(a, b) {
    return (a * b) / (Math.sqrt(a * a) * Math.sqrt(b * b));
}

async function embed(text) {
    const pipe = await loadModel();
    const output = await pipe(text, { pooling: "mean", normalize: true });
    return output.data; // Already a vector
}

async function calculate() {
    const sent1 = document.getElementById("t1").value;
    const sent2 = document.getElementById("t2").value;

    document.getElementById("output").innerText = "Loading...";

    const v1 = await embed(sent1);
    const v2 = await embed(sent2);

    // Compute cosine similarity
    let dot = 0;
    for (let i = 0; i < v1.length; i++) dot += v1[i] * v2[i];

    document.getElementById("output").innerText = dot.toFixed(4);
}
