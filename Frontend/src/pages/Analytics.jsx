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
        <h2>No data available</h2>
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

  return (
    <div className="container">
      <h1>Analytics</h1>

      {/* Graph 1 */}
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
              stroke="red"
              strokeDasharray="4 4"
              label="Threshold"
            />
            <Bar dataKey="value" fill="#ff9800" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Graph 2 */}
      <div className="result-card" style={{ marginTop: "20px" }}>
        <h3>Prediction vs True Label</h3>

        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={comparisonData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="name" />
            <YAxis domain={[0, 1]} />
            <Tooltip />
            <Bar dataKey="value" fill="#4CAF50" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      <button style={{ marginTop: "20px" }} onClick={() => navigate(-1)}>
        Back
      </button>
    </div>
  );
}

export default Analytics;