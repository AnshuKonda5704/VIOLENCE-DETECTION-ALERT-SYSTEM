
// //-----------------------------------------------------
// import React, { useState } from 'react';
// import axios from 'axios';
// import './App.css';

// function App() {
//   const [videoFile, setVideoFile] = useState(null);
//   const [result, setResult] = useState(null);
//   const [loading, setLoading] = useState(false);

//   // Handle file selection
//   const handleFileChange = (e) => {
//     setVideoFile(e.target.files[0]);
//   };

//   // Upload video and get detection results
//   const handleUpload = async () => {
//     if (!videoFile) {
//       alert("Please select a video file first.");
//       return;
//     }
//     const formData = new FormData();
//     formData.append('video', videoFile);

//     setLoading(true);
//     try {
//       // Adjust the URL to match your backend endpoint
//       const response = await axios.post('http://localhost:5000/api/detect', formData, {
//         headers: { 'Content-Type': 'multipart/form-data' },
//       });
//       setResult(response.data);
//     } catch (error) {
//       console.error("Error during detection:", error);
//       setResult({ message: "Error during detection. Please try again." });
//     } finally {
//       setLoading(false);
//     }
//   };

//   return (
//     <div className="app-container">
//       <header className="app-header">
//         <h1>Violence Detection</h1>
//         <p>Upload a video file to detect violence.</p>
//       </header>
//       <div className="upload-container">
//         <input 
//           type="file" 
//           accept="video/*" 
//           onChange={handleFileChange} 
//           className="file-input"
//         />
//         <button 
//           onClick={handleUpload} 
//           disabled={loading} 
//           className="upload-button"
//         >
//           {loading ? "Processing..." : "Upload Video"}
//         </button>
//       </div>
//       {result && (
//         <div className="result-card">
//           <h2>Detection Result</h2>
//           <p>{result.message}</p>
//         </div>
//       )}
//     </div>
//   );
// }

// export default App;
//--------------------------------------------------
import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

function App() {
  const [videoFile, setVideoFile] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  // Handle file selection
  const handleFileChange = (e) => {
    setVideoFile(e.target.files[0]);
  };

  // Upload video and get detection results
  const handleUpload = async () => {
    if (!videoFile) {
      alert("Please select a video file first.");
      return;
    }
    const formData = new FormData();
    formData.append('video', videoFile);

    setLoading(true);
    try {
      // Ensure this URL matches your backend endpoint.
      const response = await axios.post('http://localhost:5000/api/detect', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (error) {
      console.error("Error during detection:", error);
      setResult({ message: "Error during detection. Please try again." });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="app-header">
        <h1>Violence Detection</h1>
        <p>Upload a video to detect violence and view the detailed report.</p>
      </header>
      <div className="upload-container">
        <input 
          type="file" 
          accept="video/*" 
          onChange={handleFileChange} 
          className="file-input"
        />
        <button 
          onClick={handleUpload} 
          disabled={loading} 
          className="upload-button"
        >
          {loading ? "Processing..." : "Upload Video"}
        </button>
      </div>
      {result && (
        <>
          <div className="result-card">
            <h2>Detection Report</h2>
            <p><strong>Message:</strong> {result.message}</p>
            <p><strong>Upload Time:</strong> {result.upload_time}</p>
            <p><strong>Detection Time:</strong> {result.detection_time}</p>
            <p><strong>Processing Duration:</strong> {result.processing_duration} seconds</p>
            <p><strong>Violence Frames:</strong> {result.violence_frames}</p>
            <p><strong>Non-Violence Frames:</strong> {result.nonviolence_frames}</p>
          </div>
          {result.graph && (
            <div className="graph-card">
              <h2>Detection Graph</h2>
              <img
                src={`data:image/png;base64,${result.graph}`}
                alt="Detection Graph"
                className="graph-image"
              />
            </div>
          )}
        </>
      )}
    </div>
  );
}

export default App;
