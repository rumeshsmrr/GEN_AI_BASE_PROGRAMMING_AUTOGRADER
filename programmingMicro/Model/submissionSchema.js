const mongoose = require("mongoose");

const submissionSchema = new mongoose.Schema({
  assignment_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "Assignment",
    required: true,
  }, // Reference to the assignment
  student_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  }, // Reference to the student
  submitted_code: { type: String, required: true },
  grades: {
    syntax_correctness: { type: Number },
    output_match: { type: Number },
    code_quality: { type: Number },
    error_handling: { type: Number },
    boundary_conditions: { type: Number },
  },
  feedback: {
    syntax_correctness: { type: String },
    output_match: { type: String },
    code_quality: { type: String },
    error_handling: { type: String },
    boundary_conditions: { type: String },
  },
  total_score: { type: Number },
  submitted_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Submission", submissionSchema);
