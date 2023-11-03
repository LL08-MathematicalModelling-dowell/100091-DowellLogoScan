import { useEffect, useMemo, useRef, useState } from "react";
import { motion } from "framer-motion";
const Mask = ({ parentRef, maskShape, x, y }) => {
  const maskRef = useRef(null);
  const getPosition = (elementRef) => {
    let rect = elementRef.current.getBoundingClientRect();
    return {
      top: rect.top,
      left: rect.left,
      right: rect.right,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
    };
  };
  useEffect(() => {
    let maskPos = getPosition(maskRef);
    let parentPos = getPosition(parentRef);
    setChild(maskPos);
    setParent(parentPos);
  }, [x, y, maskShape]);
  const [child, setChild] = useState({});
  const [parent, setParent] = useState({});
  const mask_top = (parent.height - child.height) / 2;
  const mask_left = (parent.width - child.width) / 2;

  const mask_styles = {
    square: {
      padding: `${x}px`,
      aspectRatio: 1 / 1,
    },
    circle: {
      padding: `${x}px`,
      borderRadius: "50%",
      aspectRatio: 1 / 1,
    },
    ellipse: {
      padding: `${y}px ${x}px`,
      height: `${y}px`,
      minpadding: "100px",
      minHeight: "100px",
      borderRadius: "50%",
    },
  };
  const position = {
    border: "3px dashed rgb(203, 203, 203)",
    position: "absolute",
    top: `${mask_top}px`,
    left: `${mask_left}px`,
  };

  return (
    <motion.i
      className="fa-solid fa-plus axis-center"
      style={{ ...mask_styles[maskShape], ...position }}
      ref={maskRef}
    ></motion.i>
  );
};

export default Mask;
