from flask import Flask, request, render_template
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    pipeline
)

app = Flask(__name__)

# Load Question Generator model
qg_model_name = "iarfmoose/t5-base-question-generator"
qg_tokenizer = T5Tokenizer.from_pretrained(qg_model_name)
qg_model = T5ForConditionalGeneration.from_pretrained(qg_model_name)

# Load Question Answering model
qa_model_name = "deepset/roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)

# Generate questions
def generate_questions(context, num_questions=5):
    input_text = "generate questions: " + context
    input_ids = qg_tokenizer.encode(input_text, return_tensors="pt")
    outputs = qg_model.generate(
        input_ids=input_ids,
        max_length=128,
        num_return_sequences=num_questions,
        num_beams=5,
        early_stopping=True
    )
    questions = [qg_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return questions

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        topic = request.form.get("topic")
        grade = request.form.get("grade")
        subject = request.form.get("subject")
        num_questions = int(request.form.get("num_questions", 5))
        context = f"{topic} for grade {grade} in {subject}"

        questions = generate_questions(context, num_questions)
        quiz = []

        for i, question in enumerate(questions):
            result = qa_pipeline(question=question, context=context)
            correct_answer = result['answer'] if result['score'] > 0.3 else "Answer not confidently found"
            user_answer = request.form.get(f"user_answer_{i}", "").strip()
            is_correct = user_answer.lower() == correct_answer.lower()
            quiz.append({
                "question": question,
                "correct_answer": correct_answer,
                "user_answer": user_answer,
                "is_correct": is_correct
            })

        return render_template("home_1.html", quiz=quiz, submitted=True)

    return render_template("home_1.html", quiz=None, submitted=False)

if __name__ == '__main__':
    app.run(debug=True)
