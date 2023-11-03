import { Link } from "react-router-dom";
const User = () => {
  
  return (
    <div className="container">
      <div style={{ display: "grid", placeItems: "center", height: '80vh' }}>
        <ul className="p-0 col-4">
          <li className="py-3">
            <Link to={`/scanlogo`} className="btn btn-light col-12">
              User
            </Link>
          </li>
          <li className="py-3">
            <Link to={`/login/admin`} className="btn btn-light col-12">
              Admin
            </Link>
          </li>
        </ul>
      </div>
    </div>
  );
};

export default User;
