const axios = require("axios");
const Submission = require("../Model/Submission");
const Assignment = require("../Model/Assignment"); // Import the Assignment model

exports.evaluateAndSaveSubmission = async (req, res) => {
  try {
    const { assignment_id, student_id, answer_code, input_data } = req.body;

    if (!assignment_id || !student_id || !answer_code) {
      return res
        .status(400)
        .json({ success: false, message: "Missing required fields" });
    }

    const assignment = await Assignment.findById(req.body.assignment_id);
    if (!assignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    const reference_code = assignment.reference_code;
    const rubric = assignment.rubric;

    const flaskApiUrl = "http://127.0.0.1:5000/evaluate";
    const requestData = {
      reference_code,
      answer_code,
      input_data,
      rubric,
    };

    console.log("Request Data:", requestData);
    const flaskResponse = await axios.post(flaskApiUrl, requestData);
    const evaluationResult = flaskResponse.data;

    console.log("Evaluation Result:", evaluationResult);

    if (!evaluationResult.grades) {
      throw new Error("Grades field missing in Flask API response");
    }

    const submission = new Submission({
      assignment_id,
      student_id,
      submitted_code: answer_code,
      grades: evaluationResult.grades || {},
      total_score: evaluationResult.total_score || 0,
      detailed_results: evaluationResult.grades || {},
      code_similarity_percentage: evaluationResult.code_similarity_percentage,
      code_similarity_details: evaluationResult.code_similarity_details || {},
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

//get all submissions
exports.getAllSubmissions = async (req, res) => {
  try {
    const submissions = await Submission.find();
    res.status(200).json({ success: true, data: submissions });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get single submission by ID
exports.getSubmissionById = async (req, res) => {
  try {
    const { id } = req.params;

    const submission = await Submission.findById(id);

    if (!submission) {
      return res
        .status(404)
        .json({ success: false, message: "Submission not found" });
    }

    res.status(200).json({ success: true, data: submission });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get all submition by assignment id
exports.getAllSubmissionsByAssignment = async (req, res) => {
  try {
    const { assignment_id } = req.params;

    const submissions = await Submission.find({ assignment_id });

    res.status(200).json({ success: true, data: submissions });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get all submissions by student
exports.getAllSubmissionsByStudent = async (req, res) => {
  try {
    const { student_id } = req.params;

    const submissions = await Submission.find({ student_id });

    res.status(200).json({ success: true, data: submissions });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get single submission by ID for student
exports.getSingleSubmissionForStudent = async (req, res) => {
  try {
    const { id } = req.params;

    const submission = await Submission.findById(id);

    if (!submission) {
      return res
        .status(404)
        .json({ success: false, message: "Submission not found" });
    }

    res.status(200).json({ success: true, data: submission });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
