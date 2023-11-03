import { createSlice } from "@reduxjs/toolkit";

const initialStateValue = {
  value : null
}
const chunkSlice = createSlice({
  name: 'chunks',
  initialState: initialStateValue,
  reducers: {
    modifyChunks: (state, action)=>{
      state.value = action.payload
    }
  }
})

export const { modifyChunks } = chunkSlice.actions;
export default chunkSlice.reducer;