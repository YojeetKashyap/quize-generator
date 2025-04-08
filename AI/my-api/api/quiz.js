const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 3000;

const HUGGINGFACE_API_TOKEN = process.env.HUGGINGFACE_API_TOKEN;

// Helper to call Hugging Face models
async function callHuggingFaceModel(model, inputs) {
    const response = await axios.post(
        `https://api-inference.huggingface.co/models/${model}`,
        inputs,
        {
            headers: {
                Authorization: `Bearer ${HUGGINGFACE_API_TOKEN}`,
                'Content-Type': 'application/json'
            }
        }
    );
    return response.data;
}

// Generate questions using T5 model
async function generateQuestions(context, numQuestions = 5) {
    const inputText = "generate questions: " + context;
    const model = "iarfmoose/t5-base-question-generator";
    const result = await callHuggingFaceModel(model, { inputs: inputText, parameters: { num_return_sequences: numQuestions } });

    return result.map(r => r.generated_text || r); // HuggingFace may vary format
}

// Answer questions using QA model
async function getAnswer(question, context) {
    const model = "deepset/roberta-base-squad2";
    const result = await callHuggingFaceModel(model, { inputs: { question, context } });

    return (result.answer && result.score > 0.3) ? result.answer : "Answer not confidently found";
}

app.get('/quiz', async (req, res) => {
    const { topic, grade, subject, num_questions } = req.query;
    const numQuestions = parseInt(num_questions) || 5;

    if (!topic || !grade || !subject) {
        return res.status(400).json({ error: "Please provide 'topic', 'grade', and 'subject' as URL parameters." });
    }

    const context = `${topic} for grade ${grade} in ${subject}`;

    try {
        const questions = await generateQuestions(context, numQuestions);
        const quiz = [];

        for (const question of questions) {
            const answer = await getAnswer(question, context);
            quiz.push({ question, answer });
        }

        res.json({
            topic,
            grade,
            subject,
            context,
            quiz
        });
    } catch (error) {
        console.error(error.response?.data || error.message);
        res.status(500).json({ error: "Something went wrong." });
    }
});

app.listen(PORT, () => {
    console.log(`Quiz API running at http://localhost:${PORT}`);
});
