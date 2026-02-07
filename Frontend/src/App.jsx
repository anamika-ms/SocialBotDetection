import axios from "axios";
import { useState } from "react";

function App() {
  const [result, setResult] = useState(null);

  const handlePredict = async () => {
    try {
      const response = await axios.post("http://127.0.0.1:8000/predict", {
        structured_features: new Array(50).fill(0.5),
        graph_features: [1, 1, 1, 1, 1]
      });

      setResult(response.data);
    } catch (error) {
      console.error(error);
    }
  };

  return (
    <div style={{ padding: "40px" }}>
      <h1>Bot Detection</h1>

      <button onClick={handlePredict}>
        Predict
      </button>

      {result && (
        <div style={{ marginTop: "20px" }}>
          <p><strong>Prediction:</strong> {result.prediction}</p>
          <p><strong>Bot Probability:</strong> {result.bot_probability}</p>
          <p><strong>Threshold:</strong> {result.threshold}</p>
        </div>
      )}
    </div>
  );
}

export default App;