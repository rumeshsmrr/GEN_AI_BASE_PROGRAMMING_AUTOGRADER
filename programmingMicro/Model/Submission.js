const mongoose = require("mongoose");

const submissionSchema = new mongoose.Schema({
  assignment_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Assignment", // Reference to the Assignment model
    required: true,
  },
  student_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User", // Reference to the User model
    required: true,
  },
  submitted_code: {
    type: String,
    required: true,
  },
  grades: {
    type: Map, // Dynamic key-value structure for grades
    of: Number, // Grade values are numbers
  },
  code_similarity_percentage: { type: Number, default: 0 },
  feedback: {
    type: Map, // Dynamic key-value structure for feedback
    of: String, // Feedback values are strings
  },
  total_score: {
    type: Number, // Overall score for the submission
    required: true,
  },
  detailed_results: {
    type: Object, // Stores the entire evaluation response
  },
  submitted_at: {
    type: Date,
    default: Date.now, // Automatically records the submission time
  },
  code_similarity_details: {
    type: Map,
    of: String,
  },
});

module.exports = mongoose.model("Submission", submissionSchema);
