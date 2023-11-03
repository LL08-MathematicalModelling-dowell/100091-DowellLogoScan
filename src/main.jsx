import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./index.css";
import {createBrowserRouter, RouterProvider} from "react-router-dom";
import { configureStore } from "@reduxjs/toolkit";
import { Provider } from "react-redux";
import Login from "./pages/login";
import Home from "./pages/home";
import ScanLogo from "./pages/scanlogo";
import resultReducer from "./slices/resultSlice.js";
import chunkReducer from "./slices/chunkSLice.js";
import Upload from "./pages/upload";
import ReviewsPage from "./pages/ReviewsPage";
import User from "./pages/user";

const store = configureStore({
  reducer: {
    results: resultReducer,
    chunks: chunkReducer
  },
});

const routes = createBrowserRouter([
  {
    path: "/",
    element: <App />,
    children: [
      {
        index: true,
        element: <Home />,
      },
      {
        path: "user",
        element: <User />,
      },
      {
        path: "login/:client",
        element: <Login />,
      },
      {
        path: "scanlogo",
        element: <ScanLogo />,
      },
      {
        path: "upload/:client",
        element: <Upload />,
      },
      {
        path: "reviews/:id",
        element: <ReviewsPage />,
      },
    ],
  },
]);
ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Provider store={store}>
      <RouterProvider router={routes} />
    </Provider>
  </React.StrictMode>
);
