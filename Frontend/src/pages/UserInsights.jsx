import { useLocation, useNavigate } from "react-router-dom";

function UserInsights() {

  const location = useLocation();
  const navigate = useNavigate();

  const result = location.state?.result;

  if (!result) {
    return (
      <div className="container">
        <h2>No user data available</h2>
        <button onClick={() => navigate("/")}>Back</button>
      </div>
    );
  }

  const followers = result.followers_count || 0;
  const following = result.following_count || 0;

  const ratio = (following / (followers + 1)).toFixed(2);

  return (
    <div className="container">

      <h1>User Behavioral Insights</h1>

      <div className="analytics-card">

        <h2>Account Statistics</h2>

        <p><strong>User ID:</strong> {result.user_id}</p>
        <p><strong>Followers:</strong> {followers}</p>
        <p><strong>Following:</strong> {following}</p>
        <p><strong>Total Tweets:</strong> {result.tweet_count}</p>
        <p><strong>Follow Ratio:</strong> {ratio}</p>

      </div>


      <div className="analytics-card">

        <h2>Tweet Samples</h2>

        {result.tweets && result.tweets.length > 0 ? (
          result.tweets.map((tweet, i) => (
            <p key={i} style={{border:"1px solid #ddd",padding:"10px",marginBottom:"10px"}}>
              {tweet}
            </p>
          ))
        ) : (
          <p>No tweets available</p>
        )}

      </div>


      <div className="analytics-card">

        <h2>Follower vs Following Graph</h2>

        <div style={{
          display:"flex",
          alignItems:"end",
          height:"200px",
          gap:"40px"
        }}>

          <div>
            <div style={{
              height: `${followers/10}px`,
              width:"80px",
              background:"#4CAF50"
            }}></div>
            <p>Followers</p>
          </div>

          <div>
            <div style={{
              height: `${following/10}px`,
              width:"80px",
              background:"#f44336"
            }}></div>
            <p>Following</p>
          </div>

        </div>

      </div>


      <button
        style={{marginTop:"20px"}}
        onClick={() => navigate("/")}
      >
        Back to Home
      </button>

    </div>
  );
}

export default UserInsights;