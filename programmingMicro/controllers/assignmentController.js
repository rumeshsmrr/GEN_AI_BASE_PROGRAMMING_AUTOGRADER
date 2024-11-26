const Assignment = require("../Model/Assignment");

// Helper function to generate unique assignmentID
const generateAssignmentID = async () => {
  const now = new Date();
  const dayOfYear = Math.floor(
    (Date.UTC(now.getFullYear(), now.getMonth(), now.getDate()) -
      Date.UTC(now.getFullYear(), 0, 0)) /
      24 /
      60 /
      60 /
      1000
  );
  const year = now.getFullYear();

  // Count assignments created on the same day
  const countToday = await Assignment.countDocuments({
    created_at: {
      $gte: new Date(now.setHours(0, 0, 0, 0)),
      $lt: new Date(now.setHours(23, 59, 59, 999)),
    },
  });

  return `ASG${dayOfYear}${year}${countToday + 1}`;
};

// Create a new assignment
exports.createAssignment = async (req, res) => {
  try {
    const {
      title,
      description,
      reference_code,
      rubric,
      instructor_id,
      deadline,
    } = req.body;

    // Generate unique assignmentID
    const assignmentID = await generateAssignmentID();

    const newAssignment = new Assignment({
      assignmentID,
      title,
      description,
      reference_code,
      rubric,
      instructor_id,
      deadline,
    });

    const savedAssignment = await newAssignment.save();
    res.status(201).json({ success: true, data: savedAssignment });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// Update an assignment
exports.updateAssignment = async (req, res) => {
  try {
    const { id } = req.params; // Assignment ID
    const updatedData = req.body;

    const updatedAssignment = await Assignment.findByIdAndUpdate(
      id,
      updatedData,
      {
        new: true,
        runValidators: true,
      }
    );

    if (!updatedAssignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    res.status(200).json({ success: true, data: updatedAssignment });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

// Delete an assignment
exports.deleteAssignment = async (req, res) => {
  try {
    const { id } = req.params;

    const deletedAssignment = await Assignment.findByIdAndDelete(id);

    if (!deletedAssignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    res
      .status(200)
      .json({ success: true, message: "Assignment deleted successfully" });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get all assignments
exports.getAllAssignments = async (req, res) => {
  try {
    const assignments = await Assignment.find();
    res.status(200).json({ success: true, data: assignments });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};

//get single assignment by ID
exports.getAssignmentById = async (req, res) => {
  try {
    const { id } = req.params;

    const assignment = await Assignment.findById(id);

    if (!assignment) {
      return res
        .status(404)
        .json({ success: false, message: "Assignment not found" });
    }

    res.status(200).json({ success: true, data: assignment });
  } catch (error) {
    res.status(500).json({ success: false, error: error.message });
  }
};
