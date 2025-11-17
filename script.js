let pipelineInstance = null;
const outputDiv = document.getElementById("output");
outputDiv.innerText = "Loading model... Please wait.";

async function initEnvironment() {
    try {
        await window.transformers.env.use("webgl");
        console.log("Using WebGL backend for acceleration");
    } catch (e) {
        console.log("WebGL not available, falling back to CPU");
    }
}

async function loadModel() {
    await initEnvironment();
    pipelineInstance = await window.transformers.pipeline(
        "feature-extraction",
        "Xenova/all-MiniLM-L6-v2"
    );
    outputDiv.innerText = "Model loaded! Enter sentences and click Compute.";
}

async function embed(text) {
    const output = await pipelineInstance(text, { pooling: "mean", normalize: true });
    return output.data;
}

function cosine(v1, v2) {
    let dot = 0, norm1 = 0, norm2 = 0;
    for (let i = 0; i < v1.length; i++) {
        dot += v1[i] * v2[i];
        norm1 += v1[i] * v1[i];
        norm2 += v2[i] * v2[i];
    }
    return dot / (Math.sqrt(norm1) * Math.sqrt(norm2));
}

async function calculate() {
    const t1 = document.getElementById("t1").value.trim();
    const t2 = document.getElementById("t2").value.trim();

    if (!t1 || !t2) {
        outputDiv.innerText = "Please enter both sentences.";
        return;
    }

    outputDiv.innerText = "Computing similarity...";

    try {
        const v1 = await embed(t1);
        const v2 = await embed(t2);

        const sim = cosine(v1, v2);
        outputDiv.innerText = `Similarity: ${sim.toFixed(4)}`;
    } catch (e) {
        console.error(e);
        outputDiv.innerText = "Error computing similarity.";
    }
}

loadModel();