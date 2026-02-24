import axios from "axios";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Select from "react-select";

function Home() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/users")
      .then(res => {
        const formatted = res.data.users.map(u => ({
          value: u,
          label: u
        }));
        setUsers(formatted);
      })
      .catch(err => console.error(err));
  }, []);

  const handlePredict = () => {
    if (!selectedUser) return;
    navigate(`/result/${selectedUser.value}`);
  };

  return (
    <div className="container">

      {/* HERO SECTION */}
      <div className="hero-glow"></div>

      <h1>Social Bot Detection Dashboard</h1>

      <p className="page-description">
        AI-powered multi view bot detection system using structured behavioral signals
        and network graph embeddings enhanced by Self-Supervised Contrastive Learning.
      </p>

      {/* MAIN CARD */}
      <div className="card">

        <h3>Select User</h3>

        <Select
          options={users}
          value={selectedUser}
          onChange={setSelectedUser}
          placeholder="Search or Select User ID..."
          className="react-select-container"
          classNamePrefix="react-select"
        />

        <div style={{ textAlign: "center", marginTop: "25px" }}>
  <button onClick={handlePredict}>
    Run Detection
  </button>
</div>

      </div>

    </div>
  );
}

export default Home;