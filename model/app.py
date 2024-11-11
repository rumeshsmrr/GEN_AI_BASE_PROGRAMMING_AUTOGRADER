from flask import Flask, request, jsonify
from model import evaluate_code  # Import the evaluation function from model.py

app = Flask(__name__)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    try:
        # Get the input data (reference_code, answer_code, input_data, and rubric)
        data = request.get_json()
        reference_code = data.get('reference_code')
        answer_code = data.get('answer_code')
        input_data = data.get('input_data', "")
        rubric = data.get('rubric', {})  # User-defined rubric for weighting

        # Evaluate the code using CodeBERT and other criteria
        evaluation_result = evaluate_code(reference_code, answer_code, input_data, rubric)
        
        # Return the evaluation results as a JSON response
        return jsonify(evaluation_result)

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
