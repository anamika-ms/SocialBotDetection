import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Home from "./pages/Home";
import Result from "./pages/Result";
import Analytics from "./pages/Analytics";
import UserInsights from "./pages/UserInsights";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/result/:userId" element={<Result />} />
        <Route path="/analytics" element={<Analytics />} />
        <Route path="/insights" element={<UserInsights />} />
      </Routes>
    </Router>
  );
}

export default App;

