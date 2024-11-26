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
    type: Map, // Dynamic key-value pairs for grades
    of: Number, // Values represent the grade (Number)
  },
  feedback: {
    type: Map, // Dynamic key-value pairs for feedback
    of: String, // Values represent the feedback (String)
  },
  total_score: { type: Number },
  submitted_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("Submission", submissionSchema);
