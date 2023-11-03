import React, { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useLoaderData } from "react-router-dom";

const ReviewsPage = () => {
  const url = import.meta.env.VITE_BACKEND_API_URL;
  const [reviews, setReviews] = useState([]);
  const imageId = useParams().imageId;
  return (
    <>
      <div>
        <h1>Reviews for Image{imageId}</h1>
        <ul>
          {reviews?.map((review, index) => (
            <li key={index}>
              <p>{review.username}</p>
              <p>{review.feedback}</p>
            </li>
          ))}
        </ul>
      </div>
    </>
  );
};
export default ReviewsPage;
