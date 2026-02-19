import axios from "axios";
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

function Result() {
  const { userId } = useParams();
  const [result, setResult] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    axios.post("http://127.0.0.1:8000/predict", {
      user_id: userId
    })
      .then(res => setResult(res.data))
      .catch(err => console.error(err));
  }, [userId]);

  if (!result) {
    return (
      <div className="container">
        <h2>Loading...</h2>
      </div>
    );
  }

  return (
    <div className="container">
      <h1>Prediction Result</h1>
      <p className="page-description">
  Model inference result based on fused structured and network graph embeddings.
</p>

      <div className="result-card">
        <h2>
          Prediction:
          <span className={result.prediction === "bot" ? "bot" : "human"}>
            {result.prediction.toUpperCase()}
          </span>
        </h2>

        <p><strong>User ID:</strong> {result.user_id}</p>
        {/* <p><strong>True Label:</strong> {result.true_label}</p> */}

        <p>
          <strong>Status:</strong>
          <span className={result.correct ? "correct" : "incorrect"}>
            {result.correct ? " Correct" : " Incorrect"}
          </span>
        </p>

        {/* Probability Bar */}
        <div className="probability-bar">
          <div
            className="probability-fill"
            style={{ width: `${result.bot_probability * 100}%` }}
          ></div>
        </div>

        {/* Professional Confidence Section */}

{(() => {
  const confidence = (result.bot_probability * 100).toFixed(2);
  const threshold = (result.threshold * 100).toFixed(0);

  let riskLevel = "";
  let riskClass = "";

  if (confidence < 30) {
    riskLevel = "Low";
    riskClass = "risk-low";
  } else if (confidence < 70) {
    riskLevel = "Moderate";
    riskClass = "risk-medium";
  } else {
    riskLevel = "High";
    riskClass = "risk-high";
  }

  return (
    <>
      <p><strong>Bot Probability:</strong> {confidence}%</p>
      {/* <p><strong>Decision Threshold:</strong> {threshold}%</p> */}
      <p>
        <strong>Risk Level:</strong>
        <span className={riskClass}> {riskLevel}</span>
      </p>
    </>
  );
})()}

        <div style={{ marginTop: "20px" }}>
          <button
            style={{ marginRight: "10px" }}
            onClick={() => navigate("/")}
          >
            Back
          </button>

          <button
            onClick={() => navigate("/analytics", { state: { result } })}
          >
            View Analytics
          </button>
        </div>
      </div>
    </div>
  );
}

export default Result;