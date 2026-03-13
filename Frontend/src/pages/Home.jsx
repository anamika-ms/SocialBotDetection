import axios from "axios";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import Select from "react-select";

function Home() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState(null);
  const [modelType, setModelType] = useState("optimized"); // ✅ NEW
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

    navigate(`/result/${selectedUser.value}`, {
      state: { modelType }   // ✅ PASS MODEL
    });
  };

const modelOptions = [
  { value: "optimized", label: "Optimized (Fast)" },
  { value: "full", label: "Full Model (Best)" }
];


  return (
    <div className="container">

      <div className="hero-glow"></div>

      <h1>Social Bot Detection Dashboard</h1>

      <p className="page-description">
        AI-powered multi view bot detection system using structured behavioral signals
        and network graph embeddings enhanced by Self-Supervised Contrastive Learning.
      </p>

      <div className="card">

        <h3>Select User</h3>

        <Select
  options={users}
  value={selectedUser}
  onChange={setSelectedUser}
  placeholder="Search or Select User ID..."
  styles={{
    control: (base) => ({
      ...base,
      backgroundColor: "white",
      color: "black"
    }),
    menu: (base) => ({
      ...base,
      backgroundColor: "white"
    }),
    option: (base, state) => ({
      ...base,
      backgroundColor: state.isFocused ? "#eee" : "white",
      color: "black"
    }),
    singleValue: (base) => ({
      ...base,
      color: "black"
    })
  }}
/>

        {/* ✅ MODEL SELECT */}
        <div style={{ marginTop: "20px" }}>
         
        

            <div style={{ marginTop: "20px" }}>
    <label>Select Model:</label>

   <Select
  options={modelOptions}
  value={modelOptions.find(opt => opt.value === modelType)}
  onChange={(selected) => setModelType(selected.value)}
  placeholder="Select Model..."
  styles={{
    control: (base) => ({
      ...base,
      backgroundColor: "white",
      color: "black"
    }),
    menu: (base) => ({
      ...base,
      backgroundColor: "white"
    }),
    option: (base, state) => ({
      ...base,
      backgroundColor: state.isFocused ? "#eee" : "white",
      color: "black"
    }),
    singleValue: (base) => ({
      ...base,
      color: "black"
    })
  }}
/>
  </div>
        </div>

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