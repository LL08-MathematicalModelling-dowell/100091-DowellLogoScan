import { NavLink, useNavigate } from "react-router-dom";
import front_img from "../assets/ux-live.png";
import samantha from '../assets/js4bqgt1.png'
import './components.css'
const Navbar = () => {
  const navigate = useNavigate();
  return (
    <>
      <nav className="navbar navbar-expand-lg bg-danger shadow">
        <div className="container-fluid">
          <ul className="col-12 d-flex justify-content-evenly">
            <li className="nav-item ">
              <NavLink to="/" className="nav-link text-light">
                <i className="fa-solid fa-home"></i>
              </NavLink>
            </li>
            <li className="nav-item " onClick={()=>navigate(-1)}>
              <i className="fa-solid fa-arrow-left text-light" ></i>
            </li>
            <li className="nav-item " onClick={()=>navigate(+1)}>
                <i className="fa-solid fa-arrow-right text-light"></i>
              
            </li>
            <li className="nav-item ">
              <NavLink to="/" className="nav-link text-light">
                <i className="fa-solid fa-shield"></i>
              </NavLink>
            </li>
            <li className="nav-item ">
              <NavLink to="/" className="nav-link text-light">
                <i className="fa-solid fa-gear"></i>
              </NavLink>
            </li>
            <li className="nav-item ">
              <NavLink to="/" className="nav-link text-light">
                <img src={samantha} className="img-fluid nav-img" />
              </NavLink>
            </li>
            <li className="nav-item ">
              <NavLink to="/" className="nav-link text-light">
                <img src={front_img} className="img-fluid nav-img" />
              </NavLink>
            </li>
          </ul>
        </div>
      </nav>
    </>
  );
};

export default Navbar;
