import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "./scanlogo.css";
import placeholder from "../assets/placeholder.webp";
import { motion, useAnimation } from "framer-motion";
import Mask from "../components/mask";
import { useSelector, useDispatch } from "react-redux";
import { modify } from "../slices/resultSlice";
import { modifyChunks } from "../slices/chunkSLice";
import { Link, useNavigate } from "react-router-dom";
import Dropdown from "../components/dropdown";
const ScanLogo = () => {
  const url = import.meta.env.VITE_BACKEND_API_URL;
  const navigate = useNavigate();
  const webcamRef = useRef(null);
  const constRef = useRef(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recording, setRecording] = useState(false);
  const [recordedChunks, setRecordedChunks] = useState(null);
  const videoSource = useSelector((state) => state.chunks.value);
  const [category, setCategory] = useState({});
  const [product, setProduct] = useState({});
  const [brand, setBrand] = useState({});
  const [maskShape, setmaskShape] = useState("square");
  const [selectedCam, setSelectedCam] = useState("user");
  const [x, setX] = useState(50);
  const [y, setY] = useState(50);
  const [prompt, setPrompt] = useState(false);
  const [preview, setPreview] = useState(false);
  const scanControls = useAnimation();
  const results = useSelector((state) => state.results.value);
  const dispatch = useDispatch();

  useEffect(() => {
    if (recordedChunks) {
      let videoBlob = new Blob(recordedChunks, { type: "video/webm" });
      let videoUrl = URL.createObjectURL(videoBlob);
      sendFile();
      dispatch(modifyChunks(videoUrl));
    }
  }, [recordedChunks]);
  const scan = () => {
    const mediaStream = webcamRef.current.stream;
    const recorder = new MediaRecorder(mediaStream);
    recorder.ondataavailable = (event) => {
      if (event.data.size > 0) {
        setRecordedChunks([event.data]);
      }
    };
    setMediaRecorder(recorder);
    recorder.start();
    setRecording(true);
    scanControls.start("animated");

    setTimeout(() => {
      recorder.stop();
      setRecording(false);
      scanControls.start("normal");
    }, 10000);
  };
  const stopScanning = () => {
    mediaRecorder.stop();
  };
  const sendFile = async () => {
    const formData = new FormData();
    formData.append("video", recordedChunks[0]);
    formData.append("category", category.category);
    formData.append("product", product.product);
    formData.append("brand", brand.brand);
    const res = await fetch(`${url}VideoUploadViewFrames/`, {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    data.message && dispatch(modify(data.message));
    data.images ? setPrompt(!prompt) : null;
  };

  return (
    <>
      <div className="bg-danger">
        <ul className="d-flex justify-content-evenly">
          <li className="col-4">
            <Dropdown
              dropdown_name="categories"
              query_params={{}}
              control={setCategory}
              value={category.category}
            />
          </li>
          <li className="col-4">
            <Dropdown
              dropdown_name="products"
              query_params={category}
              control={setProduct}
              value={product.product}
            />
          </li>
          <li className="col-4">
            <Dropdown
              dropdown_name="brands"
              query_params={product}
              control={setBrand}
              value={brand.brand}
            />
          </li>
        </ul>
      </div>
      <div className="container-fluid">
        <div className="row">
          <div className="col-md-6">
            <div
              className="position-relative bg-dark"
              ref={constRef}
            >
              {preview ? (
                <video src={videoSource} controls></video>
              ) : (
                <>
                  <Webcam
                    audio={false}
                    ref={webcamRef}
                    videoConstraints={{ facingMode: selectedCam }}
                  />
                  <Mask
                    parentRef={constRef}
                    maskShape={maskShape}
                    x={x}
                    y={y}
                  />
                </>
              )}
            </div>
          </div>
          <div className="mx-auto controls-container col-md-6">
            <div className="container-fluid">
              <div className="d-flex my-3" style={{ overflowX: "scroll" }}>
                {prompt ? (
                  <>
                    <div className="alert">
                      <i>No matching logo found</i>
                      <button
                        className="btn btn-sm btn-primary"
                        onClick={() => navigate("/login/user")}
                      >
                        Upload
                      </button>
                      <button
                        className="btn btn-sm btn-secondary"
                        onClick={() => setPrompt(!prompt)}
                      >
                        Close
                      </button>
                    </div>
                  </>
                ) : (
                  <>
                    {Object.keys(results)?.map((image, index) => (
                      <Link
                        key={index}
                        to={`/reviews/${image}`}
                        className="col-4 mx-2 text-light"
                        style={{ textDecoration: "none" }}
                      >
                        <img
                          src={`${url}image/${image}`}
                          alt=""
                          className="img-fluid rounded shadow"
                          style={{ aspectRatio: 1 / 1 }}
                        />
                        {results[image].toFixed(3)}
                      </Link>
                    ))}
                    {!Object.keys(results).length && (
                      <div className=" row my-3">
                        <div className="col-4">
                          <img
                            src={placeholder}
                            alt=""
                            className="img-fluid rounded shadow"
                          />
                        </div>
                        <div className="col-4">
                          <img
                            src={placeholder}
                            alt=""
                            className="img-fluid rounded shadow"
                          />
                        </div>
                        <div className="col-4">
                          <img
                            src={placeholder}
                            alt=""
                            className="img-fluid rounded shadow"
                          />
                        </div>
                      </div>
                    )}
                  </>
                )}
              </div>
            </div>
            <div className="row">
              <div className="col-md-1">
                <div className="row">
                  <button
                    className={`btn btn-sm btn-${
                      recording ? "info" : "success"
                    } me-3`}
                    onClick={() => scan()}
                    title="Scan"
                  >
                    <motion.i
                      variants={{
                        animated: {
                          scale: [1, 0.7, 1, 0.7, 1, 0.7, 1, 0.7, 1],
                        },
                        normal: { scale: 1 },
                      }}
                      animate={scanControls}
                      initial={{ scale: 1 }}
                      transition={{ type: "tween", duration: 10 }}
                      className="fa-solid fa-magnifying-glass"
                    ></motion.i>
                  </button>
                  <button
                    className="btn btn-sm btn-dark me-3"
                    onClick={() =>
                      setSelectedCam(
                        selectedCam == "user" ? "environment" : "user"
                      )
                    }
                    title="Switch camera"
                  >
                    <i className="fa-solid fa-camera-rotate"></i>
                  </button>
                  <button
                    className="btn btn-sm btn-secondary me-3"
                    onClick={() => {
                      setPreview(false);
                    }}
                    title="Rescan"
                  >
                    <i className="fa-solid fa-arrows-rotate"></i>
                  </button>
                  <button
                    className="btn btn-sm btn-primary me-3"
                    onClick={() =>
                      videoSource ? setPreview(true) : setPreview(false)
                    }
                    title="Preview"
                  >
                    <i className="fa-solid fa-eye"></i>
                  </button>
                </div>
              </div>
              <div className="col-10">
                <div className="row g-0">
                  <div className="col">
                    <label htmlFor="shapes">Shapes:</label>
                  </div>
                  <div className="col-9">
                    <select
                      name="shapes"
                      id="shapes"
                      value={maskShape}
                      onChange={(e) => setmaskShape(e.target.value)}
                    >
                      <option value="square">Square</option>
                      <option value="circle">Circle</option>
                      <option value="ellipse">Eclipse</option>
                    </select>
                    <div className="col-12 my-2">
                      <div className="div">
                        <label htmlFor="x_axis">X-axis</label>
                      </div>
                      <div className="div">
                        <input
                          type="range"
                          name="x_axis"
                          id="x_axis"
                          className="col-6"
                          value={x}
                          min={50}
                          max={130}
                          onChange={(e) => setX(e.target.value)}
                        />
                      </div>
                    </div>
                    <div className="col-12">
                      {maskShape == "ellipse" && (
                        <>
                          <div className="div">
                            <label htmlFor="y_axis">Y-axis</label>
                          </div>
                          <div className="div">
                            <input
                              type="range"
                              name="y_axis"
                              id="y_axis"
                              className="col-6"
                              value={y}
                              min={50}
                              max={130}
                              onChange={(e) => setY(e.target.value)}
                            />
                          </div>
                        </>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default ScanLogo;
