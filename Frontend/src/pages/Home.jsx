import axios from "axios";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

function Home() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState("");
  const [manualUser, setManualUser] = useState("");
  const navigate = useNavigate();

  useEffect(() => {
    axios.get("http://127.0.0.1:8000/users")
      .then(res => setUsers(res.data.users))
      .catch(err => console.error(err));
  }, []);

  const handlePredict = (userId) => {
    if (!userId) return;
    navigate(`/result/${userId}`);
  };

  return (
    <div className="container">
      <h1>Bot Detection Dashboard</h1>

      <div className="card">
        <h3>Select User</h3>
        <select
          value={selectedUser}
          onChange={(e) => setSelectedUser(e.target.value)}
        >
          <option value="">-- Select User --</option>
          {users.map((user, index) => (
            <option key={index} value={user}>{user}</option>
          ))}
        </select>

        <button onClick={() => handlePredict(selectedUser)}>
          Predict
        </button>

        {/* <h3>Or Search Manually</h3>
        <input
          type="text"
          placeholder="Enter User ID"
          value={manualUser}
          onChange={(e) => setManualUser(e.target.value)}
        />
        <button onClick={() => handlePredict(manualUser)}>
          Predict
        </button> */}
      </div>
    </div>
  );
}

export default Home;