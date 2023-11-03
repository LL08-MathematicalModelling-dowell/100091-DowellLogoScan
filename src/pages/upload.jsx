import React, { useRef, useState, useEffect } from "react";
import Webcam from "react-webcam";
import "./scanlogo.css";
import { useSelector, useDispatch } from "react-redux";
import { useParams } from "react-router-dom";

const Upload = () => {
  const url = import.meta.env.VITE_BACKEND_API_URL;
  const webcamRef = useRef(null);
  const fileInputRef = useRef(null);
  const [category, setCategory] = useState("");
  const [product, setProduct] = useState("");
  const [brand, setBrand] = useState("");
  const [selectedCam, setSelectedCam] = useState("user");
  const [imageData, setImageData] = useState(null);
  const [selectedImage, setSelectedImage] = useState(null);
  const [message, setMessage] = useState("");
  const client = useParams().client
  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedImage(URL.createObjectURL(file));
      setImageData(file);
    }
  };

  const capture = () => {
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      setSelectedImage(imageSrc);
      // Convert data URL to Blob
      fetch(imageSrc)
        .then((res) => res.blob())
        .then((blob) => setImageData(blob));
    }
  };
  const clearCapture = () => {
    setSelectedImage(null);
    setImageData(null);
    setMessage(null);
    setCategory("");
    setProduct("");
    setBrand("");
  };

  const sendFile = async () => {
    const formData = new FormData();
    formData.append("image", imageData);
    formData.append("category", category?.category);
    formData.append("product", product?.product);
    formData.append("brand", brand?.brand);
    formData.append("flag", client);
    const res = await fetch(`${url}logo-upload-image/`, {
      method: "POST",
      body: formData,
    });
    if (res.status == 200) {
      const data = await res.json();
      setMessage(data.message);
    }
  };
  const handleFileInput = () => {
    fileInputRef.current.click();
  };
  const handleForm = (event, func) => {
    let value = event.target.value;
    func(value);
  };
  return (
    <>
      <div className="container mt-3">
        {message && (
          <div className="border rounded alert d-flex justify-content-between">
            <div>
              <p>{message}</p>
            </div>
            <div className="col-2 d-flex justify-content-evenly">
              <button
                className="btn btn-sm btn-secondary"
                onClick={clearCapture}
              >
                Continue
              </button>
              <button className="btn btn-sm btn-success" onClick={clearCapture}>
                Finish
              </button>
            </div>
          </div>
        )}

        <div className="row">
          <div className="col-md-6">
            {selectedImage ? (
              <div className="img-fluid">
                <img
                  src={selectedImage}
                  alt=""
                  className="img-fluid"
                  style={{ aspectRatio: 3 / 2 }}
                />
              </div>
            ) : (
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                videoConstraints={{ facingMode: selectedCam }}
              />
            )}
          </div>

          <div className="col-md-6">
            <div className="container-fluid">
              <div className="row">
                <div className="col">
                  <div className="d-flex justify-content-between">
                    <button
                      className="btn btn-sm btn-primary"
                      onClick={capture}
                      title="Capture"
                    >
                      <i className="fa-solid fa-camera"></i>
                    </button>
                    <button
                      className="btn btn-sm btn-dark"
                      onClick={handleFileInput}
                      title="Gallery"
                    >
                      <i className="fa-solid fa-image"></i>
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
                      className="btn btn-sm btn-secondary"
                      title="Recapture"
                      onClick={clearCapture}
                    >
                      <i className="fa-solid fa-arrows-rotate"></i>
                    </button>
                  </div>
                </div>
                <div className="col-md-10">
                  <div className="row">
                    <div className="py-2">
                      <input
                        type="text"
                        name="category"
                        id="category"
                        placeholder="Category"
                        className="form-control"
                        value={category}
                        onChange={(e) => handleForm(e, setCategory)}
                      />
                    </div>
                    <div className="py-2">
                      <input
                        type="text"
                        name="product"
                        id="product"
                        placeholder="Product"
                        className="form-control"
                        value={product}
                        onChange={(e) => handleForm(e, setProduct)}
                      />
                    </div>
                    <div className="py-2">
                      <input
                        type="text"
                        name="brand"
                        id="brand"
                        placeholder="Brand"
                        className="form-control"
                        value={brand}
                        onChange={(e) => handleForm(e, setBrand)}
                      />
                    </div>
                    <div className="py-2">
                      <input
                        ref={fileInputRef}
                        type="file"
                        name="image"
                        id="image"
                        placeholder="image"
                        className="form-control"
                        style={{ display: "none" }}
                        onChange={(e) => handleFileChange(e)}
                      />
                    </div>
                    <button
                      className="btn btn-sm btn-success col"
                      onClick={sendFile}
                      title="Upload"
                    >
                      Upload <i className="fa-solid fa-upload"></i>
                    </button>
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

export default Upload;
