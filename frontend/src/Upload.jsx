import { useRef, useState } from "react";
import { uploadCsv } from "./api";

export default function Upload({ onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const inputRef = useRef(null);

  async function handleFile(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith(".csv")) {
      setError("Only .csv files are supported");
      return;
    }
    setLoading(true);
    setProgress(0);
    setError(null);
    try {
      const data = await uploadCsv(file, setProgress);
      onUploaded(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  const classes = [
    "upload-zone",
    "fade-in-up",
    dragging && "upload-zone--dragging",
    loading && "upload-zone--loading",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div
      className={classes}
      onDragOver={(e) => {
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={() => setDragging(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragging(false);
        handleFile(e.dataTransfer.files?.[0]);
      }}
      onClick={() => inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        hidden
        onChange={(e) => handleFile(e.target.files?.[0])}
      />
      <div className="upload-zone__mark">PULSE</div>
      <div className="upload-zone__status">
        {loading && <span className="spinner" />}
        {loading ? (progress < 100 ? `UPLOADING... ${progress}%` : "PROCESSING...") : "DROP A CSV OR CLICK TO BROWSE"}
      </div>
      {loading && (
        <div
          className={`upload-zone__progress${progress >= 100 ? " upload-zone__progress--indeterminate" : ""}`}
          role="progressbar"
          aria-valuenow={progress}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div className="upload-zone__progress-fill" style={progress < 100 ? { width: `${progress}%` } : undefined} />
        </div>
      )}
      <div className="upload-zone__hint">Any CSV — no schema required. Analyzed entirely in memory.</div>
      {error && <div className="upload-zone__error">{error}</div>}
    </div>
  );
}
