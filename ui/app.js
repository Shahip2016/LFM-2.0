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
    });
});
