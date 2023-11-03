import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";

const Login = () => {
  const url = import.meta.env.VITE_BACKEND_API_URL;
  const formInput = { username: "", password: "" };
  const navigate = useNavigate();
  const [user, setUser] = useState(formInput);
  const [error, setError] = useState(null);
  const client = useParams().client
  useEffect(() => {
    let errorTime = setTimeout(() => {
      setError(null);
    }, 5000);
    () => {
      clearTimeout(errorTime);
    };
  }, [error]);
  const handleSubmit = async (e) => {
    e.preventDefault();
    let formData = new FormData();
    formData.append("username", user["username"]);
    formData.append("password", user["password"]);
    let res = await fetch(`${url}login/`, {
      method: "POST",
      body: formData,
    });
    let data = await res.json();
    if (data.authenticated) {
      navigate(`/upload/${client}`);
    } else if (!data.authenticated) {
      setUser(formInput);
      setError("Wrong credentials");
    }
  };
  return (
    <div className="container-fluid form-container">
      {error && (
            <p className="mt-3 text-light">{error}</p>
      )}
      <form action="">
        <div className="pt-4 col-12">
          <h4 className="text-light">Login</h4>
        </div>
        <div className="py-3 col-12">
          <input
            type="text"
            name="username"
            id="username"
            placeholder="Username"
            className="form-control"
            value={user.username}
            onChange={(e) => setUser({ ...user, username: e.target.value })}
            required
          />
        </div>
        <div className="py-3 col-12">
          <input
            type="password"
            name="password"
            id="password"
            placeholder="Password"
            className="form-control"
            value={user.password}
            onChange={(e) => setUser({ ...user, password: e.target.value })}
            required
          />
        </div>
        <div className="py-3 col-12">
          <input
            type="submit"
            value="Login"
            className="btn form-control btn-secondary"
            onClick={(e) => handleSubmit(e)}
          />
        </div>
      </form>
    </div>
  );
};

export default Login;
