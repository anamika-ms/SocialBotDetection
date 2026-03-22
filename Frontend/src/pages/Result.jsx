import axios from "axios";
import { useEffect, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";

function Result() {

  const { userId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();

  // ✅ GET MODEL TYPE FROM HOME
  const modelType = location.state?.modelType || "optimized";

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(true);


  useEffect(() => {

    if (!userId) return;

    axios.post("http://127.0.0.1:8000/predict", {
      user_id: userId,
      model_type: modelType   // ✅ IMPORTANT FIX
    })
    .then(res => {
      setResult(res.data);
      setLoading(false);
    })
    .catch(err => {
      console.error(err);
      alert("Server error. Please check backend.");
      setLoading(false);
    });

  }, [userId, modelType]);


  if (loading) {
    return (
      <div className="container">
        <h2>Loading prediction...</h2>
      </div>
    );
  }


  if (!result) {
    return (
      <div className="container">
        <h2>No result available</h2>
        <button onClick={() => navigate("/")}>
          Back
        </button>
      </div>
    );
  }


  return (
    <div className="container">

      <h1>Prediction Result</h1>

      <p className="page-description">
        Model inference result based on fused structured, graph and text embeddings.
      </p>


      <div className="result-card">

        <h2>
          Prediction:
          <span className={result.prediction === "bot" ? "bot" : "human"}>
            {" "}
            {result.prediction.toUpperCase()}
          </span>
        </h2>


        <p>
          <strong>User ID:</strong> {result.user_id}
        </p>

        {/* ✅ NEW: SHOW MODEL */}
        <p>
          <strong>Model Used:</strong> {result.model_used}
        </p>


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


        {/* Risk + Confidence */}
        {(() => {

          const confidence = (result.bot_probability * 100).toFixed(2);

          let riskLevel = "";
          let riskClass = "";

          if (confidence < 30) {
            riskLevel = "Low";
            riskClass = "risk-low";
          }
          else if (confidence < 70) {
            riskLevel = "Moderate";
            riskClass = "risk-medium";
          }
          else {
            riskLevel = "High";
            riskClass = "risk-high";
          }

          return (
            <>
              <p>
                <strong>Bot Probability:</strong> {confidence}%
              </p>

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

          {/* <button
            onClick={() => navigate("/insights", { state: { result } })}
          >
            User Insights
          </button> */}

        </div>

      </div>

    </div>
  );
}

export default Result;