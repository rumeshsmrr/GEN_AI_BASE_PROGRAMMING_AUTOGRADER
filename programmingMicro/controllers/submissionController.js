const axios = require("axios");
const Submission = require("../Model/Submission");
const Assignment = require("../Model/Assignment"); // Import the Assignment model

exports.evaluateAndSaveSubmission = async (req, res) => {
  try {
    const { assignment_id, student_id, answer_code, input_data, rubric } =
      req.body;

    if (!assignment_id || !student_id || !answer_code) {
      return res
        .status(400)
        .json({ success: false, message: "Missing required fields" });
    }

    // Fetch the reference_code from the database using assignment_id
    const assignment = await Assignment.findById(assignment_id);
    if (!assignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    const reference_code = assignment.reference_code;

    // Prepare data for the Flask API
    const flaskApiUrl = "http://127.0.0.1:5000/evaluate";
    const requestData = {
      reference_code,
      answer_code,
      input_data,
      rubric,
    };

    // Call the Flask API for evaluation
    const flaskResponse = await axios.post(flaskApiUrl, requestData);

    // Extract evaluation result from Flask API response
    const evaluationResult = flaskResponse.data;

    // Save the evaluation result in the database
    const submission = new Submission({
      assignment_id,
      student_id,
      submitted_code: answer_code,
      grades: evaluationResult.detailed_results,
      total_score: evaluationResult.final_score,
      detailed_results: evaluationResult.detailed_results,
    });

    await submission.save();

    res.status(201).json({
      success: true,
      message: "Submission evaluated and saved successfully",
      data: submission,
    });
  } catch (error) {
    console.error("Error during evaluation:", error.message);
    res.status(500).json({
      success: false,
      error: error.message,
    });
  }
};
