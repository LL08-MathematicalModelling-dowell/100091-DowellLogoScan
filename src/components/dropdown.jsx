import { useEffect, useState } from "react";

const Dropdown = ({ dropdown_name, query_params, control, value }) => {
  const url = import.meta.env.VITE_BACKEND_API_URL;
  const [data, setData] = useState([]);
  const getDropdownData = async () => {
    const queryString = Object.keys(query_params)
      .map(
        (key) =>
          `${encodeURIComponent(key)}=${encodeURIComponent(query_params[key])}`
      )
      .join("&");

    let res = await fetch(`${url}dropdown-menu-data/?${queryString}`);
    if (res.status == 200) {
      let dat = await res.json();
      setData([null, ...dat[dropdown_name]]);
    }
  };
  useEffect(() => {
    getDropdownData();
  }, [query_params]);
  const handleDropdown = (e) => {
    let { name, value } = e.target;
    if (name == "categories") {
      control({ category: value });
    } else if (name == "products") {
      control({ product: value });
    } else if (name == "brands") {
      control({ brand: value });
    }
  };
  return (
    <select
      className="bg-danger col-12"
      name={dropdown_name}
      onChange={(e) => handleDropdown(e)}
      value={value}
    >
      {data?.map((item, index) => (
        <option key={index} value={item}>
          {item}
        </option>
      ))}
    </select>
  );
};

export default Dropdown;
