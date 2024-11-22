from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load CodeT5 (or other similar model) for prompt-based evaluations
model_name = "Salesforce/codet5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Load CodeT5 for classification tasks (e.g., syntax checking)
classification_model_name = "microsoft/codebert-base"
classification_tokenizer = AutoTokenizer.from_pretrained(classification_model_name)
classification_model = AutoModelForSequenceClassification.from_pretrained(classification_model_name)

# Generate prompt for a criterion using CodeT5
def generate_prompt_for_criterion(criterion):
    """
    Generates a question for the LLM to evaluate the Java program based on a criterion.
    """
    input_text = f"Create a question to evaluate Java code for the following criterion: {criterion}."
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Generate the prompt
    output_ids = model.generate(input_ids["input_ids"], max_length=50, num_return_sequences=1)
    generated_prompt = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return generated_prompt

# Syntax error evaluation
def evaluate_syntax_errors(code_snippet):
    """
    Uses CodeBERT to check if the syntax of the code is correct.
    """
    prompt = f"Check the syntax of the following Java code:\n{code_snippet}"
    inputs = classification_tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = classification_model(**inputs)
    logits = outputs.logits
    predicted_label = torch.argmax(logits).item()
    return 100 if predicted_label == 1 else 0

# Validate method existence
def validate_method_existence(code_snippet, method_name):
    """
    Validates if a specific method exists in the Java class.
    """
    return 100 if f"{method_name}(" in code_snippet else 0

# Evaluate code similarity using TF-IDF and cosine similarity
def evaluate_code_similarity(ref_code, ans_code):
    vectorizer = TfidfVectorizer().fit_transform([ref_code, ans_code])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim * 100

# Compile and execute Java code
def run_java_code(code, input_data, filename):
    """
    Compiles and runs the given Java code.
    """
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code)

    # Compile Java code
    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        return f"Compilation Error: {compile_process.stderr}"

    # Run Java program
    run_process = subprocess.run(
        ["java", filename.replace(".java", "")],
        input=input_data,
        text=True,
        capture_output=True,
    )
    if run_process.returncode != 0:
        return f"Runtime Error: {run_process.stderr}"

    return run_process.stdout

# Evaluate output match
def evaluate_output_match(reference_code, answer_code, input_data=""):
    ref_output = run_java_code(reference_code, input_data, "Reference")
    ans_output = run_java_code(answer_code, input_data, "Answer")
    return compare_outputs(ref_output, ans_output)

# Compare outputs
def compare_outputs(ref_output, ans_output):
    """
    Compares outputs line by line and calculates a match percentage.
    """
    ref_lines = ref_output.splitlines()
    ans_lines = ans_output.splitlines()
    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref, ans in zip(ref_lines, ans_lines) if ref == ans)
    return (matching_lines / total_lines) * 100 if total_lines > 0 else 0.0

# Evaluate dynamic criteria with LLM
def evaluate_dynamic_criteria_with_llm(code_snippet, criteria, rubric):
    """
    Evaluates multiple criteria dynamically using CodeT5.
    """
    results = {}
    for criterion in criteria:
        weight = rubric.get(criterion, 0)
        if weight == 0:
            continue

        prompt = generate_prompt_for_criterion(criterion)
        complete_prompt = f"{prompt}\n{code_snippet}"

        inputs = tokenizer(complete_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model.generate(inputs["input_ids"], max_length=50)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Assume response contains "yes"/"no" or similar and score accordingly
        results[criterion] = 100 if "yes" in response.lower() else 50
        results[criterion] *= weight / 100
    return results

# Main evaluation function
def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    """
    Main function to evaluate a Java program against a rubric.
    """
    # Static evaluations
    syntax_score = evaluate_syntax_errors(answer_code)
    output_match = evaluate_output_match(reference_code, answer_code, input_data)
    code_similarity = evaluate_code_similarity(reference_code, answer_code)

    # Initialize results
    results = {
        "syntax_score": syntax_score * rubric.get("syntax_correctness", 0) / 100,
        "output_match_percentage": output_match * rubric.get("output_match", 0) / 100,
        "code_similarity_percentage": code_similarity * rubric.get("code_similarity", 0) / 100,
    }

    # Dynamic evaluations
    dynamic_criteria = [key for key in rubric if key not in ["syntax_correctness", "output_match", "code_similarity"]]
    dynamic_scores = evaluate_dynamic_criteria_with_llm(answer_code, dynamic_criteria, rubric)
    results.update(dynamic_scores)

    # Final score
    final_score = min(100, sum(results.values()))
    return {
        "final_score": round(final_score, 2),
        "detailed_results": results,
    }
