import { Link } from "react-router-dom";
import home_logo from "../assets/js4bqgt1.png";
const Home = () => {
  return (
    <>
      <div className="container">
        <div className="mx-auto mt-5" style={{ width: "fit-content" }}>
          <div className="col-4 mx-auto">
            <Link to={`/user`}>
              <img src={home_logo} alt="" className="img-fluid rounded-pill" />
            </Link>
          </div>
        </div>
      </div>
    </>
  );
};

export default Home;
