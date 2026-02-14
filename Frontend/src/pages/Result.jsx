import axios from "axios";
import { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer
} from "recharts";

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

  const chartData = [
    {
      name: "Bot Probability",
      value: result.bot_probability
    }
  ];

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

        {/* Graph Section */}
        <div style={{ marginTop: "30px" }}>
          <h3>Probability vs Threshold</h3>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <ReferenceLine
                y={result.threshold}
                stroke="red"
                strokeDasharray="4 4"
                label="Threshold"
              />
              <Bar dataKey="value" fill="#ff9800" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        <button
          style={{ marginTop: "20px" }}
          onClick={() => navigate("/")}
        >
          Back
        </button>
      </div>
    </div>
  );
}

export default Result;