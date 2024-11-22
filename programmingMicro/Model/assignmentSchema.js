const mongoose = require("mongoose");

const assignmentSchema = new mongoose.Schema({
  assignmentID: { type: String, required: true },
  title: { type: String, required: true },
  description: { type: String, required: true },
  reference_code: { type: String, required: true },
  rubric: {
    syntax_correctness: { type: Number, required: true },
    output_match: { type: Number, required: true },
    code_quality: { type: Number, required: true },
    error_handling: { type: Number, required: true },
    boundary_conditions: { type: Number, required: true },
  },
  instructor_id: {
    type: mongoose.Schema.Types.ObjectId,
    ref: "User",
    required: true,
  }, // Reference to the instructor
  created_at: { type: Date, default: Date.now },
  deadline: { type: Date, required: true },
});

module.exports = mongoose.model("Assignment", assignmentSchema);
