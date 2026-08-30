import { Routes, Route } from "react-router-dom";
import HomePage from "./HomePage";
import BuildPage from "./BuildPage";
import ReviewPage from "./ReviewPage";
import StoryPage from "./StoryPage";

function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/build" element={<BuildPage />} />
      <Route path="/review" element={<ReviewPage />} />
      <Route path="/story" element={<StoryPage />} />
    </Routes>
  );
}

export default App;
