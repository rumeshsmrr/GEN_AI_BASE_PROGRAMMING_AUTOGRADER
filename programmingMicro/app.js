const express = require("express");
const assignmentRoutes = require("./routes/assignmentRoutes");

module.exports = (app) => {
  app.use(express.json()); // Middleware to parse JSON

  // Routes
  app.use("/api/assignments", assignmentRoutes);

  // Example root route
  app.get("/", (req, res) => {
    res.send("API is running...");
  });
};
