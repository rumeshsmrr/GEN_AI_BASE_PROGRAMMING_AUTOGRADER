const mongoose = require("mongoose");

const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  role: { type: String, required: true, enum: ["student", "instructor"] }, // Either 'student' or 'instructor'
  created_at: { type: Date, default: Date.now },
});

module.exports = mongoose.model("User", userSchema);
