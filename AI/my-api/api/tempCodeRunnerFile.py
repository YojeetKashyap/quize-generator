from flask import Flask, request, jsonify
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

@app.route('/quiz', methods=['GET'])
def quiz_json():
    topic = request.args.get("topic")
    grade = request.args.get("grade")
    subject = request.args.get("subject")
    num_questions = int(request.args.get("num_questions", 5))

    if not all([topic, grade, subject]):
        return jsonify({"error": "Please provide 'topic', 'grade', and 'subject' as URL parameters."}), 400

    context = f"{topic} for grade {grade} in {subject}"
    questions = generate_questions(context, num_questions)

    quiz = []
    for question in questions:
        result = qa_pipeline(question=question, context=context)
        answer = result['answer'] if result['score'] > 0.3 else "Answer not confidently found"
        quiz.append({"question": question, "answer": answer})

    return jsonify({
        "topic": topic,
        "grade": grade,
        "subject": subject,
        "context": context,
        "quiz": quiz
    })

if __name__ == '__main__':
    app.run(debug=True)
