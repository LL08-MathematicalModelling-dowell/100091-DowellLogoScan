import { createSlice } from "@reduxjs/toolkit";

const initialStateValue = {
  value : {}
}
const resultSlice = createSlice({
  name : 'results',
  initialState: initialStateValue,
  reducers: {
    modify : (state, action)=>{
      state.value = action.payload
    }
  }
})

export const { modify } = resultSlice.actions;
export default resultSlice.reducer;