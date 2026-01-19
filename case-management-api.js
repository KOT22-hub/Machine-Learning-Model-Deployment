const express = require('express');
const ort = require('onnxruntime-node');
const app = express();
const port = 3000
app.use(express.json());

const Model = './case_predictor.onnx';
async function runModel(inputData) {
    try {
        const session = await ort.InferenceSession.create(Model);
        const inputValues = Float32Array.from(inputData);
        const tensor = new ort.Tensor('float32', inputValues, [1, 5]);
        
        const feeds = { float_input: tensor };
        const results = await session.run(feeds);

        const labelTensor = results.output_label || results.label;
        
        // 2. Try to find the probabilities
        const probTensor = results.output_probability || results.probabilities;

        if (!labelTensor) {
            throw new Error("Could not find prediction labels in model output.");
        }

        return {
            is_resolved: Number(labelTensor.data[0]),
            // Use optional chaining (?.) to prevent crashes if probs are missing
            confidence: probTensor ? Array.from(probTensor.data) : "Not available"
        };
    } catch (error) {
        console.error("Inference Error:", error);
        throw error;
    }
}
app.post('/predict', async (req, res) => {
    const {age, risk_score, years_exp, workload, priority} = req.body;
const features = [
        parseFloat(age), 
        parseFloat(risk_score), 
        parseFloat(years_exp), 
        parseFloat(workload), 
        parseFloat(priority)
    ];
    try {
        const result = await runModel(features);
        res.json({ success: true, prediction: result });
    } catch (err) {
        res.status(500).json({ success: false, error: err.message });
    }
});

app.listen(port, () => {
    console.log(`Server is running on port ${port}`);


});