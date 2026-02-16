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

      <div className="result-card">
        <h2>
          Prediction:
          <span className={result.prediction === "bot" ? "bot" : "human"}>
            {result.prediction.toUpperCase()}
          </span>
        </h2>

        <p><strong>User ID:</strong> {result.user_id}</p>
        <p><strong>True Label:</strong> {result.true_label}</p>

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

        <p><strong>Bot Probability:</strong> {result.bot_probability}</p>
        <p><strong>Threshold:</strong> {result.threshold}</p>

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