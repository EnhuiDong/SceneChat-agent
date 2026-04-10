import { Routes, Route } from "react-router-dom";
import HomePage from "./HomePage";
import StoryPage from "./StoryPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/story" element={<StoryPage />} />
    </Routes>
  );
}

export default App;