const express = require("express");
const assignmentRoutes = require("./routes/assignmentRoutes");
const submissionRoutes = require("./routes/submissionRoutes");

module.exports = (app) => {
  app.use(express.json()); // Middleware to parse JSON

  // Routes
  app.use("/api/assignments", assignmentRoutes);
  app.use("/api/submissions", submissionRoutes);

  // Example root route
  app.get("/", (req, res) => {
    res.send("API is running...");
  });
};
