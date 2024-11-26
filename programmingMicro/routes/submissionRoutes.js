const express = require("express");
const router = express.Router();
const {
  evaluateAndSaveSubmission,
} = require("../controllers/submissionController");

// Route to evaluate and save a submission
router.post("/evaluate", evaluateAndSaveSubmission);

module.exports = router;
