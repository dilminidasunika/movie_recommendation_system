import { useState } from "react";
import { getPosterUrl } from "../poster";

export default function MoviePoster({ path, title, className = "" }) {
  const [hasError, setHasError] = useState(false);
  const url = getPosterUrl(path);

  if (!url || hasError) {
    return (
      <div
        className={`card-poster placeholder ${className}`.trim()}
        aria-label={title ? `No poster for ${title}` : "No poster available"}
      >
        <svg
          className="poster-fallback-icon"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          aria-hidden="true"
        >
          <rect x="2" y="3" width="20" height="18" rx="2" />
          <line x1="7" y1="3" x2="7" y2="21" />
          <line x1="17" y1="3" x2="17" y2="21" />
          <line x1="2" y1="12" x2="22" y2="12" />
          <line x1="2" y1="7" x2="7" y2="7" />
          <line x1="2" y1="17" x2="7" y2="17" />
          <line x1="17" y1="17" x2="22" y2="17" />
          <line x1="17" y1="7" x2="22" y2="7" />
        </svg>
        <span className="poster-fallback-text">No poster</span>
      </div>
    );
  }

  return (
    <div className={`card-poster ${className}`.trim()}>
      <img
        src={url}
        alt={title ? `${title} poster` : "Movie poster"}
        className="poster-img"
        loading="lazy"
        onError={() => setHasError(true)}
      />
    </div>
  );
}
