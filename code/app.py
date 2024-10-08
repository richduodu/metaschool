from dotenv import load_dotenv
import os
from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import gemini

# Load environment variables from the .env file
load_dotenv()

# Access the API keys
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
gemini_api_key = os.getenv('GEMINI_API_KEY')

# Initialize the Flask app
app = Flask(__name__)

# Class for the AI tutor that interacts with students and helps them learn.
class AITutor:
    def __init__(self, model_name="gpt-3.5-turbo"):
        # Load the Hugging Face pipeline for text generation
        self.qa_pipeline = pipeline('text2text-generation', model=model_name, api_key=huggingface_api_key)
        self.gemini_model = None
    
    def explain_concept(self, question):
        """Takes a student's question and explains the concept."""
        response = self.qa_pipeline(question)
        return response[0]['generated_text']

    def interactive_solver(self, problem_description):
        """Guides the student step-by-step through solving the problem."""
        steps = [
            "What is the given mass?", 
            "What is the given acceleration?", 
            "What is the formula for force?"
        ]
        responses = []

        for step in steps:
            responses.append(f"AI Tutor: {step}")
        return responses

    def guardrails(self, answer):
        """Ensures AI follows safe and constructive behavior by not giving full answers."""
        if "the answer is" in answer:
            return "Let's break it down further. How would you approach this?"
        return answer

    def use_gemini(self, input_text):
        """Uses Gemini SDK to offload inference to an optimized NVIDIA GPU."""
        if not self.gemini_model:
            self.gemini_model = gemini.load_model('path_to_model', device='gpu')
        return self.gemini_model.predict(input_text)

# Initialize the AI Tutor instance
tutor = AITutor()

# Flask Routes

@app.route('/')
def index():
    """Render the main UI page."""
    return render_template('index.html')

@app.route('/ask', methods=["POST"])
def ask_question():
    """Handles the POST request for asking a question."""
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Question not provided"}), 400
    # Explain the concept
    response = tutor.explain_concept(question)
    # Apply guardrails to prevent giving away answers
    response = tutor.guardrails(response)
    return jsonify({"response": response})

@app.route('/solve', methods=["POST"])
def solve_problem():
    """Handles the POST request for solving a problem interactively."""
    data = request.json
    problem_description = data.get("problem", "")
    if not problem_description:
        return jsonify({"error": "Problem description not provided"}), 400
    # Interactive problem-solving session
    steps = tutor.interactive_solver(problem_description)
    return jsonify({"steps": steps})

# Run the Flask server
if __name__ == "__main__":
    app.run(debug=True)
