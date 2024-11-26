require("dotenv").config(); // Load environment variables from .env
const express = require("express");
const connectDB = require("./config/dbConfig");

// Initialize the Express app
const app = express();

// Connect to MongoDB
connectDB();

// Import routes from app.js
const setupApp = require("./app");
setupApp(app); // Pass the app instance to configure routes and middleware

// Start the server
const PORT = process.env.PORT || 5001;
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
