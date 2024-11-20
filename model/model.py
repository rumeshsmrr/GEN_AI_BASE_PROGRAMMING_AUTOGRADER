from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
import torch
import subprocess
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-trained tokenizer and model for CodeBERT
tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Load GPT-2 for prompt generation
nlp_tokenizer = AutoTokenizer.from_pretrained("gpt2")
nlp_model = AutoModelForCausalLM.from_pretrained("gpt2")

# Generate prompt for a criterion using GPT-2
def generate_prompt_for_criterion(criterion):
    input_text = f"Create a question to evaluate Java code for the following criterion: {criterion}."
    input_ids = nlp_tokenizer.encode(input_text, return_tensors='pt')

    # Generate text
    output_ids = nlp_model.generate(input_ids, max_length=50, num_return_sequences=1)
    generated_text = nlp_tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Clean up generated text
    prompt = generated_text.replace(input_text, "").strip()
    return prompt

# Evaluate code dynamically using GPT-2-generated prompts and CodeBERT
def evaluate_dynamic_criteria_with_llm(code_snippet, criteria, rubric):
    results = {}
    for criterion in criteria:
        weight = rubric.get(criterion, 0)
        if weight == 0:
            continue  # Skip criteria with no weight

        prompt = generate_prompt_for_criterion(criterion)
        complete_prompt = f"{prompt}\n{code_snippet}"

        # Tokenize and get output from CodeBERT
        inputs = tokenizer(complete_prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()

        # Assign score (100 = good, 50 = bad) and apply weight
        score = 100 if predicted_label == 1 else 50
        weighted_score = score * (weight / 100)
        results[criterion] = weighted_score

    return results

# Explicitly validate the existence of specific methods
def validate_method_existence(code_snippet, method_name):
    if f"{method_name}(" in code_snippet:
        return 100  # Full marks if the method exists
    return 0  # No marks if the method does not exist

# Evaluate syntax errors
def evaluate_syntax_errors(code_snippet):
    try:
        prompt = f"Evaluate the syntax of the following Java code:\n{code_snippet}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        predicted_label = torch.argmax(logits).item()
        return 100 if predicted_label == 1 else 0
    except Exception as e:
        print(f"Error in syntax evaluation: {e}")
        return 0

# Java code execution (compilation and runtime)
def run_java_code(code, input_data, filename):
    filename = f"{filename}.java"
    with open(filename, "w") as f:
        f.write(code)

    # Compile the Java code
    compile_process = subprocess.run(["javac", filename], capture_output=True, text=True)
    if compile_process.returncode != 0:
        return f"Compilation Error: {compile_process.stderr}"

    # Run the Java program
    run_process = subprocess.run(
        ["java", filename.replace(".java", "")], capture_output=True, text=True, input=input_data
    )
    if run_process.returncode != 0:
        return f"Runtime Error: {run_process.stderr}"

    return run_process.stdout

# Evaluate output match
def evaluate_output_match(reference_code, answer_code, input_data=""):
    ref_output = run_java_code(reference_code, input_data, "Reference")
    ans_output = run_java_code(answer_code, input_data, "Answer")
    matching_percentage = compare_outputs(ref_output, ans_output)
    return matching_percentage

# Compare outputs
def compare_outputs(ref_output, ans_output):
    ref_lines = ref_output.splitlines()
    ans_lines = ans_output.splitlines()
    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref_line, ans_line in zip(ref_lines, ans_lines) if ref_line == ans_line)

    if total_lines == 0:
        return 100.0 if ref_output == ans_output else 0.0
    return (matching_lines / total_lines) * 100

# Evaluate code similarity
def evaluate_code_similarity(ref_code, ans_code):
    vectorizer = TfidfVectorizer().fit_transform([ref_code, ans_code])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim * 100

# Main evaluation function
def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    # Evaluate static criteria
    syntax_score = evaluate_syntax_errors(answer_code)
    output_match_percentage = evaluate_output_match(reference_code, answer_code, input_data)
    code_similarity_percentage = evaluate_code_similarity(reference_code, answer_code)

    # Identify dynamic criteria
    static_criteria = ["syntax_correctness", "output_match", "code_similarity"]
    dynamic_criteria = [key for key in rubric.keys() if key not in static_criteria]

    # Initialize results
    evaluation_results = {
        "syntax_score": syntax_score * rubric.get("syntax_correctness", 0) / 100,
        "output_match_percentage": output_match_percentage * rubric.get("output_match", 0) / 100,
        "code_similarity_percentage": code_similarity_percentage * rubric.get("code_similarity", 0) / 100,
    }

    # Evaluate dynamic criteria
    for criterion in dynamic_criteria:
        weight = rubric.get(criterion, 0)
        if "method" in criterion.lower():  # Explicit validation for methods
            method_name = criterion.split()[-1].strip("()")
            score = validate_method_existence(answer_code, method_name)
            evaluation_results[criterion] = score * (weight / 100)
        else:
            # Use GPT-2 and CodeBERT for other criteria
            dynamic_scores = evaluate_dynamic_criteria_with_llm(answer_code, [criterion], rubric)
            evaluation_results.update(dynamic_scores)

    # Calculate the final score
    final_score = sum(evaluation_results.values())
    final_score = min(100, final_score)  # Cap at 100

    return {
        "final_score": round(final_score, 2),
        "detailed_results": evaluation_results,
    }
