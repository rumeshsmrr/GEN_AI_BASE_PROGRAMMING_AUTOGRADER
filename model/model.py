import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess
import javalang

# Load CodeT5 for prompt-based evaluations
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load CodeBERT for syntax checking
classification_model_name = "microsoft/codebert-base"
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

def normalize_code(code):
    """
    Normalizes the code by removing extra whitespace and formatting inconsistencies.
    """
    return "\n".join(line.strip() for line in code.splitlines() if line.strip())

def evaluate_syntax_errors_with_javalang(code_snippet):
    """
    Validates Java syntax using javalang.
    Returns the number of syntax errors found.
    """
    try:
        javalang.parse.parse(code_snippet)
        return 0  # No syntax errors
    except javalang.parser.JavaSyntaxError as e:
        print(f"Syntax error detected by javalang: {e}")
        return 1  # Count as one syntax error
    except Exception as e:
        print(f"Unexpected error in javalang: {e}")
        return 1  # Count as one syntax error for unexpected issues

def evaluate_syntax_errors(code_snippet, rubric_weight):
    """
    Uses CodeBERT and javalang to validate syntax.
    Reduces 0.5 marks per syntax error.
    """
    max_score = rubric_weight  # Max score for syntax correctness
    error_penalty = 0.5  # Deduction per syntax error

    # Start with CodeBERT-based syntax checking
    try:
        prompt = f"Is the syntax of this Java code correct? Yes or No:\n{code_snippet}"
        inputs = classification_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = classification_model(**inputs)
        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()

        # If CodeBERT detects syntax errors, apply one penalty
        if predicted_label != 1:
            max_score -= error_penalty
    except Exception as e:
        print(f"CodeBERT syntax check failed: {e}")
        max_score -= error_penalty

    # Fall back to javalang for more granular error detection
    syntax_errors = evaluate_syntax_errors_with_javalang(code_snippet)
    max_score -= syntax_errors * error_penalty

    # Ensure score is not negative
    return max(0, max_score)

def evaluate_code_similarity(ref_code, ans_code):
    """
    Evaluate code similarity using normalized TF-IDF and cosine similarity.
    """
    ref_code_normalized = normalize_code(ref_code)
    ans_code_normalized = normalize_code(ans_code)

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

def evaluate_dynamic_criteria_with_llm(code_snippet, criteria, rubric):
    """
    Evaluates multiple criteria dynamically using CodeT5.
    """
    results = {}
    for criterion in criteria:
        weight = rubric.get(criterion, 0)
        if weight == 0:
            continue

        try:
            prompt = f"Evaluate the following Java code for the criterion: {criterion}\n{code_snippet}"
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = model.generate(inputs["input_ids"], max_length=50)
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            results[criterion] = 100 if "yes" in response.lower() else 50
            results[criterion] *= weight / 100
        except Exception as e:
            results[criterion] = 0
            print(f"Error evaluating criterion {criterion}: {e}")
    return results

def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    """
    Main function to evaluate a Java program against a rubric. If code similarity is 100%, final score is set to 100.
    """
    # Static evaluations
    syntax_score = evaluate_syntax_errors(answer_code, rubric.get("syntax_correctness", 10))
    output_match = evaluate_output_match(reference_code, answer_code, input_data)
    code_similarity = evaluate_code_similarity(reference_code, answer_code)

    # Initialize results
    results = {
        "syntax_score": syntax_score,
        "output_match_percentage": output_match * rubric.get("output_match", 0) / 100,
    }

    # Add dynamic rubric evaluations
    dynamic_criteria = [key for key in rubric if key not in ["syntax_correctness", "output_match", "code_similarity"]]
    dynamic_scores = evaluate_dynamic_criteria_with_llm(answer_code, dynamic_criteria, rubric)
    results.update(dynamic_scores)

    # Compute final score
    if code_similarity == 100:  # If code similarity is perfect, set the final score to 100
        final_score = 100
    else:
        final_score = min(100, sum(results.values()))  # Calculate based on rubric

    return {
        "final_score": round(final_score, 2),
        "detailed_results": results,
        "code_similarity_percentage": round(code_similarity, 2)  
    }
