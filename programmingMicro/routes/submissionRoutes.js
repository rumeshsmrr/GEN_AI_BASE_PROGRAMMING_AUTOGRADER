const express = require("express");
const router = express.Router();
const {
  evaluateAndSaveSubmission,
  getAllSubmissions,
  getSubmissionById,
  getAllSubmissionsByAssignment,
  getAllSubmissionsByStudent,
  getSingleSubmissionForStudent,
} = require("../controllers/submissionController");

// Route to evaluate and save a submission
router.post("/evaluate", evaluateAndSaveSubmission);
router.get("/getAll", getAllSubmissions);
router.get("/:id", getSubmissionById);
router.get("/assignment/:assignment_id", getAllSubmissionsByAssignment);
router.get("/student/:student_id", getAllSubmissionsByStudent);
router.get("/student/:_id", getSingleSubmissionForStudent);
module.exports = router;
