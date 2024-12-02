import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import subprocess
import javalang
import re

# Load CodeT5 for prompt-based evaluations
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load CodeBERT for syntax checking
classification_model_name = "microsoft/codebert-base"  # Updated to use CodeBERT
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)


def normalize_code(code):
    """
    Normalizes the code by removing extra whitespace and formatting inconsistencies.
    """
    return "\n".join(line.strip() for line in code.splitlines() if line.strip())


def detect_syntax_errors_with_javalang(code_snippet):
    """
    Validates Java syntax using javalang.
    Returns a list of syntax errors.
    """
    errors = []
    try:
        javalang.parse.parse(code_snippet)
    except javalang.parser.JavaSyntaxError as e:
        errors.append(str(e))
    except Exception as e:
        errors.append(f"Unexpected error: {str(e)}")
    return errors


def detect_syntax_errors_with_compiler(code_snippet, filename="Temp"):
    """
    Compiles Java code to identify syntax errors.
    """
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code_snippet)

    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        return compile_process.stderr.splitlines()  # Return syntax errors
    return []  # No errors


def evaluate_syntax_errors(code_snippet, rubric_weight):
    """
    Identifies syntax errors and calculates a score.
    Reduces 0.5 marks per syntax error found.
    """
    max_score = rubric_weight
    error_penalty = 0.5  # Deduction per syntax error
    all_errors = []

    # Detect errors with javalang
    javalang_errors = detect_syntax_errors_with_javalang(code_snippet)
    all_errors.extend(javalang_errors)

    # Detect errors with Java compiler
    compiler_errors = detect_syntax_errors_with_compiler(code_snippet)
    all_errors.extend(compiler_errors)

    # Syntax checking with CodeBERT
    try:
        prompt = f"Is the syntax of this Java code correct? Yes or No:\n{code_snippet}"
        inputs = classification_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()

        # If CodeBERT detects syntax errors, apply one penalty
        if predicted_label != 1:  # Assuming label 1 means "correct"
            all_errors.append("CodeBERT: Detected syntax issues.")
            max_score -= error_penalty
    except Exception as e:
        print(f"CodeBERT syntax check failed: {e}")
        max_score -= error_penalty

    # Calculate final score
    max_score -= len(all_errors) * error_penalty
    max_score = max(0, max_score)  # Ensure the score is not negative

    return max_score, all_errors


def evaluate_code_similarity(ref_code, ans_code):
    """
    Evaluate code similarity using normalized TF-IDF and cosine similarity.
    """
    ref_code_normalized = normalize_code(ref_code)
    ans_code_normalized = normalize_code(ans_code)

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer().fit_transform([ref_code_normalized, ans_code_normalized])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim * 100


def run_java_code(code, input_data="", filename="Temp"):
    """
    Compiles and runs Java code, handling input data.
    """
    filename = f"{filename}.java"
    try:
        with open(filename, "w") as f:
            f.write(code)

        # Compile Java code
        compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
        if compile_process.returncode != 0:
            return f"Compilation Error: {compile_process.stderr.strip()}"

        # Run Java program
        run_process = subprocess.run(
            ["java", filename.replace(".java", "")],
            input=input_data,
            text=True,
            capture_output=True
        )
        if run_process.returncode != 0:
            return f"Runtime Error: {run_process.stderr.strip()}"

        return run_process.stdout.strip()
    except Exception as e:
        return f"Error: {str(e)}"


def compare_outputs(ref_output, ans_output):
    """
    Compares outputs line by line and calculates a match percentage.
    """
    ref_lines = [line.strip() for line in ref_output.splitlines() if line.strip()]
    ans_lines = [line.strip() for line in ans_output.splitlines() if line.strip()]

    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref, ans in zip(ref_lines, ans_lines) if ref == ans)

    return (matching_lines / total_lines) * 100 if total_lines > 0 else 0.0


def evaluate_output_match(reference_code, answer_code, input_data=""):
    """
    Compares the output of reference and answer code executions.
    """
    ref_output = run_java_code(reference_code, input_data, "Reference")
    ans_output = run_java_code(answer_code, input_data, "Answer")
    return compare_outputs(ref_output, ans_output)


def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    """
    Main function to evaluate a Java program against a rubric.
    """
    # Static evaluations
    syntax_score, syntax_errors = evaluate_syntax_errors(answer_code, rubric.get("syntax_correctness", 10))
    output_match = evaluate_output_match(reference_code, answer_code, input_data)
    code_similarity = evaluate_code_similarity(reference_code, answer_code)

    # Initialize results
    results = {
        "syntax_score": syntax_score,
        "output_match_percentage": output_match * rubric.get("output_match", 0) / 100,
        "code_similarity_percentage": code_similarity,  # Include raw code similarity
    }

    # Compute final score
    final_score = min(100, sum(value for key, value in results.items() if isinstance(value, (int, float))))

    return {
        "final_score": round(final_score, 2),
        "detailed_results": results,
        "syntax_errors": syntax_errors,  # Include the list of syntax errors for debugging
    }
