const express = require("express");
const router = express.Router();
const assignmentController = require("../controllers/assignmentController");

// Routes for assignment management
router.post("/create", assignmentController.createAssignment); // Create assignment
router.put("/:id", assignmentController.updateAssignment); // Update assignment
router.delete("/:id", assignmentController.deleteAssignment); // Delete assignment
router.get("/getAll", assignmentController.getAllAssignments); // Get all assignments

router.get("/allForStudent", assignmentController.getAllAssignmentForStudent); // Get an assignment by ID
router.get("/singleForStudent/:id", assignmentController.getSingleForStudent); // Get an assignment by ID

router.get("/:id", assignmentController.getAssignmentById); // Get an assignment by ID

module.exports = router;
