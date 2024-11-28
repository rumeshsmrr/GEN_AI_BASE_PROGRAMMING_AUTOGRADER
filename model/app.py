from flask import Flask, request, jsonify
from model import evaluate_code

app = Flask(__name__)

@app.route("/evaluate", methods=["POST"])
def evaluate():
    try:
        data = request.get_json()
        reference_code = data.get("reference_code")
        answer_code = data.get("answer_code")
        rubric = data.get("rubric")
        test_cases = data.get("input_data", "")

        # Evaluate the code
        evaluation_result = evaluate_code(reference_code, answer_code, test_cases, rubric)
        return jsonify(evaluation_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
