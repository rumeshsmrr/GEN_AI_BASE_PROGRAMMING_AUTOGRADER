const mongoose = require("mongoose");

const assignmentSchema = new mongoose.Schema({
  assignmentID: { type: String, required: true },
  title: { type: String, required: true },
  description: { type: String, required: true },
  reference_code: { type: String, required: true },
  rubric: {
    type: Map, // Dynamic key-value structure for grades
    of: Number, // Grade values are numbers
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
