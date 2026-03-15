document.addEventListener('DOMContentLoaded', () => {
    const generateBtn = document.getElementById('generate-btn');
    const tokenRange = document.getElementById('token-range');
    const tokenValue = document.getElementById('token-value');
    const tempRange = document.getElementById('temp-range');
    const tempValue = document.getElementById('temp-value');
    const outputContent = document.getElementById('output-content');
    const promptInput = document.getElementById('prompt-input');

    // Update range values
    tokenRange.addEventListener('input', (e) => tokenValue.textContent = e.target.value);
    tempRange.addEventListener('input', (e) => tempValue.textContent = e.target.value);

    const dummyResponses = [
        "LFM 2.0 architecture utilizes Gated Short Convolutions to achieve high efficiency in processing sequence data. This allows for linear-time complexity while maintaining strong representational power.",
        "The GQA (Grouped Query Attention) mechanism in LFM 2.0 balances the performance of Multi-Head Attention with the memory efficiency of Multi-Query Attention.",
        "Liquid Foundation Models are designed to adapt their internal state dynamically, making them ideal for long-context applications and real-time processing tasks."
    ];

    async function streamOutput(text) {
        outputContent.innerHTML = '';
        const cursor = document.createElement('span');
        cursor.className = 'cursor';
        
        const words = text.split(' ');
        for (let i = 0; i < words.length; i++) {
            const wordSpan = document.createElement('span');
            wordSpan.textContent = words[i] + ' ';
            wordSpan.style.opacity = '0';
            wordSpan.style.transform = 'translateY(5px)';
            wordSpan.style.transition = 'all 0.1s ease';
            
            outputContent.appendChild(wordSpan);
            outputContent.appendChild(cursor);
            
            // Trigger reflow
            wordSpan.offsetHeight;
            
            wordSpan.style.opacity = '1';
            wordSpan.style.transform = 'translateY(0)';
            
            await new Promise(resolve => setTimeout(resolve, 50 + Math.random() * 50));
        }
    }

    generateBtn.addEventListener('click', async () => {
        if (!promptInput.value.trim()) return;

        generateBtn.disabled = true;
        generateBtn.textContent = 'Generating...';
        
        const response = dummyResponses[Math.floor(Math.random() * dummyResponses.length)];
        await streamOutput(response);

        generateBtn.disabled = false;
        generateBtn.textContent = 'Generate Response';
    }); // end of generateBtn click listener

    // Training Metrics Simulation
    let currentLoss = 4.5;
    let currentThroughput = 12000;

    function updateMetrics() {
        // Simulate training progress
        currentLoss = Math.max(0.1, currentLoss - (Math.random() * 0.05));
        currentThroughput = 12000 + (Math.random() * 1000 - 500);

        const metrics = {
            loss: currentLoss.toFixed(4),
            throughput: Math.floor(currentThroughput).toLocaleString(),
            tokens: (currentThroughput * 1.2).toFixed(0),
            step: Math.floor(Date.now() / 1000) % 100000
        };

        const lossVal = document.getElementById('loss-val');
        const throughputVal = document.getElementById('throughput-val');
        const stepVal = document.getElementById('step-val');

        if (lossVal) lossVal.textContent = metrics.loss;
        if (throughputVal) throughputVal.textContent = metrics.throughput + ' tokens/sec';
        if (stepVal) stepVal.textContent = metrics.step;
    }

    // Start simulation if dashboard exists
    if (document.getElementById('loss-val') || document.getElementById('throughput-val')) {
        setInterval(updateMetrics, 1000);
        updateMetrics(); // Initial call
        console.log("Dashboard simulation started.");
    }
});

// Feature: Clear Output functionality
document.addEventListener('DOMContentLoaded', () => {
    const clearBtn = document.getElementById('clear-btn');
    if (clearBtn) {
        clearBtn.addEventListener('click', () => {
            const outputContent = document.getElementById('output-content');
            if (outputContent) outputContent.innerHTML = '';
        });
    }
});
