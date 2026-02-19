import { useLocation, useNavigate } from "react-router-dom";
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

function Analytics() {
  const location = useLocation();
  const navigate = useNavigate();
  const result = location.state?.result;

  if (!result) {
    return (
      <div className="container">
        <h1>Analytics</h1>
        <p className="page-description">
          No analytics data available. Please run a prediction first.
        </p>
        <button onClick={() => navigate("/")}>Go Home</button>
      </div>
    );
  }

  const probabilityData = [
    {
      name: "Bot Probability",
      value: result.bot_probability
    }
  ];

  const comparisonData = [
    {
      name: "Prediction",
      value: result.prediction === "bot" ? 1 : 0
    },
    {
      name: "True Label",
      value: result.true_label === "bot" ? 1 : 0
    }
  ];
  <h1>Analytics</h1>
  return (
    <div className="container">
      
      <h1>Analytics</h1>

      <p className="page-description">
        Visual interpretation of model confidence and comparison against ground truth labels.
      </p>
        

        
      <div className="analytics-grid">

        {/* LEFT GRAPH */}
        <div className="result-card">
          <h3>Probability vs Threshold</h3>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={probabilityData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <ReferenceLine
                y={result.threshold}
                stroke="#ff4d6d"
                strokeDasharray="4 4"
                label="Threshold"
              />
              <Bar dataKey="value" fill="#00f5ff" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* RIGHT GRAPH */}
        <div className="result-card">
          <h3>Prediction vs True Label</h3>

          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis domain={[0, 1]} />
              <Tooltip />
              <Bar dataKey="value" fill="#8f00ff" />
            </BarChart>
          </ResponsiveContainer>
        </div>

      </div>

      <div style={{ marginTop: "40px", textAlign: "center" }}>
        <button onClick={() => navigate(-1)}>
          Back to Result
        </button>
      </div>
    </div>
  );
}

export default Analytics;