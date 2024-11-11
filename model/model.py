from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import subprocess

# Load the pre-trained CodeBERT model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
model = RobertaForSequenceClassification.from_pretrained("microsoft/codebert-base")

# Function to evaluate code using CodeBERT
def evaluate_code_with_llm(code_snippet, criterion="error_handling"):
    if criterion == "error_handling":
        prompt = f"Does the following code handle errors properly?\n{code_snippet}"
    elif criterion == "boundary_conditions":
        prompt = f"Does the following code handle boundary conditions properly?\n{code_snippet}"
    elif criterion == "code_quality":
        prompt = f"Is the following code written with good quality? Is it well-commented, well-indented, and clear?\n{code_snippet}"
    elif criterion == "code_robustness":
        prompt = f"Does the following code have adequate robustness? Does it handle edge cases properly?\n{code_snippet}"
    elif criterion == "oop_principles":
        prompt = f"Does the following Java code follow OOP principles like encapsulation, inheritance, and polymorphism?\n{code_snippet}"

    # Tokenize and generate output from CodeBERT model
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get prediction (1 = "good", 0 = "bad")
    logits = outputs.logits
    predicted_label = torch.argmax(logits).item()

    # Return numerical score (100 if good, 50 if bad)
    return 100 if predicted_label == 1 else 50

# Syntax Error Evaluation
def evaluate_syntax_errors(code_snippet):
    prompt = f"Evaluate the syntax of the following Java code:\n{code_snippet}"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_label = torch.argmax(logits).item()

    # Syntax error evaluation score
    return 100 if predicted_label == 1 else 0

# Output Matching Evaluation (compare outputs of reference and answer code)
def evaluate_output_match(reference_code, answer_code, input_data=""):
    ref_output = run_java_code(reference_code, input_data, "Reference")
    ans_output = run_java_code(answer_code, input_data, "Answer")
    matching_percentage = compare_outputs(ref_output, ans_output)
    return matching_percentage

# Code Similarity Evaluation (using cosine similarity)
def evaluate_code_similarity(reference_code, answer_code):
    return compare_code_similarity(reference_code, answer_code)

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
    run_process = subprocess.run(["java", filename.replace(".java", "")], capture_output=True, text=True, input=input_data)
    if run_process.returncode != 0:
        return f"Runtime Error: {run_process.stderr}"

    return run_process.stdout

# Output comparison
def compare_outputs(ref_output, ans_output):
    ref_lines = ref_output.splitlines()
    ans_lines = ans_output.splitlines()
    total_lines = max(len(ref_lines), len(ans_lines))
    matching_lines = sum(1 for ref_line, ans_line in zip(ref_lines, ans_lines) if ref_line == ans_line)
    
    if total_lines == 0:
        return 100.0 if ref_output == ans_output else 0.0
    return (matching_lines / total_lines) * 100

# Code similarity using cosine similarity
def compare_code_similarity(ref_code, ans_code):
    vectorizer = TfidfVectorizer().fit_transform([ref_code, ans_code])
    vectors = vectorizer.toarray()
    cosine_sim = cosine_similarity(vectors)[0, 1]
    return cosine_sim * 100

# Overall evaluation function
def evaluate_code(reference_code, answer_code, input_data="", rubric={}):
    # 1. Syntax Evaluation
    syntax_score = evaluate_syntax_errors(answer_code)
    
    # 2. Evaluate output matching
    output_match_percentage = evaluate_output_match(reference_code, answer_code, input_data)

    # 3. Code similarity evaluation
    code_similarity_percentage = evaluate_code_similarity(reference_code, answer_code)

    # 4. CodeBERT-based evaluations for other criteria
    error_handling = evaluate_code_with_llm(answer_code, "error_handling")
    boundary_conditions = evaluate_code_with_llm(answer_code, "boundary_conditions")
    code_quality = evaluate_code_with_llm(answer_code, "code_quality")
    code_robustness = evaluate_code_with_llm(answer_code, "code_robustness")
    oop_principles = evaluate_code_with_llm(answer_code, "oop_principles")

    
    # Check if code similarity is 100% and assign full marks if so
    if code_similarity_percentage == 100:
        code_similarity_percentage = 100
    else:
        code_similarity_percentage = code_similarity_percentage * rubric.get('code_similarity', 0) / 100


    # Combine results into a dictionary (adjusted for code similarity)
    evaluation_results = {
        "syntax_score": syntax_score * rubric.get('syntax_correctness', 0) / 100,
        "output_match_percentage": output_match_percentage * rubric.get('output_match', 0) / 100,
        "code_similarity_percentage": code_similarity_percentage,
        "error_handling": (rubric.get('error_handling', 0) / 100) * (error_handling),
        "boundary_conditions": (rubric.get('boundary_conditions', 0) / 100) * (boundary_conditions),
        "code_quality": (rubric.get('code_quality', 0) / 100) * (code_quality),
        "code_robustness": (rubric.get('code_robustness', 0) / 100) * (code_robustness),
        "oop_principles": (rubric.get('oop_principles', 0) / 100) * (oop_principles)
    }

    # Calculate final score using the rubric (user-defined weights)
    final_score = 0
    final_score += syntax_score * rubric.get('syntax_correctness', 0) / 100
    final_score += output_match_percentage * rubric.get('output_match', 0) / 100
    final_score += code_similarity_percentage
    # Add the CodeBERT-based evaluation results (using user-defined weights)
    final_score += (rubric.get('error_handling', 0) / 100) * (error_handling)
    final_score += (rubric.get('boundary_conditions', 0) / 100) * (boundary_conditions)
    final_score += (rubric.get('code_quality', 0) / 100) * (code_quality)
    final_score += (rubric.get('code_robustness', 0) / 100) * (code_robustness)
    final_score += (rubric.get('oop_principles', 0) / 100) * (oop_principles)

    if code_similarity_percentage == 100:
        final_score = 100
    else:
        # Ensure the score is within 100
        final_score = min(100, final_score)

    

    # Return the final score along with detailed results
    return {
        "final_score": round(final_score, 2),
        "detailed_results": evaluation_results
    }

