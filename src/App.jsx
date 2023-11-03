import React from "react";
import { Outlet } from "react-router-dom";
import Navbar from "./components/navbar";
import Footer from "./components/footer";
function App() {
  
  return (
    <div style={{height: '100vh'}} className="position-relative bg-danger">
      <Navbar />
      <div className="mt-0">
        <Outlet />
      </div>
    </div>
  );
}

export default App;
