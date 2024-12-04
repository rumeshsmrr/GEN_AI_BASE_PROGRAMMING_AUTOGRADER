import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import subprocess
import javalang
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CodeBERT for syntax validation and semantic similarity
codebert_name = "microsoft/codebert-base"
codebert_tokenizer = AutoTokenizer.from_pretrained(codebert_name)
codebert_model = AutoModelForSequenceClassification.from_pretrained(codebert_name)

# Load CodeT5 for dynamic evaluations
codet5_name = "Salesforce/codet5-small"
codet5_tokenizer = AutoTokenizer.from_pretrained(codet5_name)
codet5_model = AutoModelForSeq2SeqLM.from_pretrained(codet5_name)

def normalize_code(code):
    """
    Strict normalization to remove all formatting inconsistencies.
    """
    return "\n".join(line.strip() for line in code.splitlines() if line.strip()).strip()

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
    Compiles Java code to identify syntax errors, ignoring file name issues.
    """
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code_snippet)

    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        errors = compile_process.stderr.splitlines()
        # Filter out file name issues
        filtered_errors = [
            error for error in errors if "should be declared in a file named" not in error
        ]
        return filtered_errors  # Return remaining errors
    return []  # No errors

def evaluate_syntax(code_snippet, rubric_weight):
    """
    Combines prompt-based and rule-based syntax evaluations.
    """
    max_score = rubric_weight
    error_penalty = 0.5  # Deduction per error
    errors = []

    # Rule-based syntax validation
    compiler_errors = detect_syntax_errors_with_compiler(code_snippet)
    if compiler_errors:
        errors.extend(compiler_errors)
        max_score -= len(compiler_errors) * error_penalty

    javalang_errors = detect_syntax_errors_with_javalang(code_snippet)
    if javalang_errors:
        errors.extend(javalang_errors)
        max_score -= len(javalang_errors) * error_penalty

    return max(0, max_score), errors

def evaluate_code_similarity(ref_code, ans_code):
    """
    Combines TF-IDF and CodeBERT for code similarity evaluation.
    """
    ref_code_normalized = normalize_code(ref_code)
    ans_code_normalized = normalize_code(ans_code)

    # Step 1: Exact match check
    if ref_code_normalized == ans_code_normalized:
        return 100.0, 100.0, 100.0

    # Step 2: TF-IDF Similarity
    vectorizer = TfidfVectorizer().fit_transform([ref_code_normalized, ans_code_normalized])
    vectors = vectorizer.toarray()
    tfidf_similarity = cosine_similarity(vectors)[0, 1] * 100

    # Step 3: CodeBERT Similarity (if TF-IDF similarity > threshold)
    if tfidf_similarity > 50:  # Use a threshold to save computational cost
        inputs = codebert_tokenizer(
            f"Determine similarity:\nCode 1:\n{ref_code}\nCode 2:\n{ans_code}",
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        )
        with torch.no_grad():
            outputs = codebert_model(**inputs)
        codebert_similarity = torch.softmax(outputs.logits, dim=1)[0][1].item() * 100  # Confidence for "similar"
    else:
        codebert_similarity = 0.0  # Skip deeper evaluation if TF-IDF similarity is too low

    # Aggregate the results
    final_similarity = (tfidf_similarity + codebert_similarity) / 2
    return round(tfidf_similarity, 2), round(codebert_similarity, 2), round(final_similarity, 2)

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

def evaluate_dynamic_criteria(code_snippet, criteria, rubric):
    """
    Evaluate multiple criteria dynamically using CodeT5.
    """
    results = {}
    for criterion in criteria:
        weight = rubric.get(criterion, 0)
        if weight == 0:
            continue

        prompt = f"""
        Evaluate the following Java code for the criterion: {criterion}.
        Respond with 'Yes' if the criterion is met, otherwise respond with 'No'.

        Code:
        \"\"\"{code_snippet}\"\"\".
        """
        try:
            inputs = codet5_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
            with torch.no_grad():
                outputs = codet5_model.generate(inputs["input_ids"], max_length=50)
            response = codet5_tokenizer.decode(outputs[0], skip_special_tokens=True)

            results[criterion] = 100 if "yes" in response.lower() else 50
            results[criterion] *= weight / 100
        except Exception as e:
            results[criterion] = 0
            print(f"CodeT5 error for {criterion}: {e}")

    return results

def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    """
    Main function to evaluate Java code.
    """
    # Syntax evaluation
    syntax_score, syntax_errors = evaluate_syntax(answer_code, rubric.get("syntax_correctness", 10))

    # Output matching
    output_match = compare_outputs(
        run_java_code(reference_code, input_data, "Reference"),
        run_java_code(answer_code, input_data, "Answer"),
    )

    # Code similarity using hybrid method
    tfidf_similarity, codebert_similarity, final_similarity = evaluate_code_similarity(reference_code, answer_code)

    # Dynamic criteria evaluation
    dynamic_criteria = [key for key in rubric if key not in ["syntax_correctness", "output_match", "code_similarity"]]
    dynamic_scores = evaluate_dynamic_criteria(answer_code, dynamic_criteria, rubric)

    # Handle perfect similarity case
    if final_similarity == 100.0:
        final_score = 100.0
        results = {key: rubric[key] for key in rubric.keys()}
        syntax_errors = []
    else:
        # Aggregated results
        results = {
            "syntax_correctness": syntax_score,
            "output_match_percentage": output_match * rubric.get("output_match", 0) / 100,
            **dynamic_scores,
        }
        final_score = min(100, sum(value for value in results.values() if isinstance(value, (int, float))))

    return {
        "final_score": round(final_score, 2),
        "grades": results,
        "total_score": round(final_score, 2),
        "code_similarity_details": {
            "TF-IDF Similarity": tfidf_similarity,
            "CodeBERT Similarity": codebert_similarity,
            "Final Aggregated Similarity": final_similarity,
        },
        "code_similarity_percentage": final_similarity,
        "syntax_errors": [] if final_similarity == 100.0 else syntax_errors,
    }
