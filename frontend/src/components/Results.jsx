import { useState } from "react";
import { hasFilters } from "../query";

const tagName = (t) => (typeof t === "string" ? t : t?.name ?? "");

function getPosterUrl(posterPath) {
  if (!posterPath || typeof posterPath !== "string") return null;
  const trimmed = posterPath.trim();
  if (!trimmed) return null;
  if (trimmed.startsWith("http://") || trimmed.startsWith("https://")) {
    return trimmed;
  }
  const cleanPath = trimmed.replace(/^\/+/, "");
  return `https://image.tmdb.org/t/p/w500/${cleanPath}`;
}

function MoviePoster({ path, title }) {
  const [hasError, setHasError] = useState(false);
  const url = getPosterUrl(path);

  if (!url || hasError) {
    return (
      <div className="card-poster placeholder" aria-label={title ? `No poster for ${title}` : "No poster available"}>
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
    <div className="card-poster">
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

function ScoreBar({ label, value, kind }) {
  const has = typeof value === "number" && !Number.isNaN(value);
  return (
    <div className="score-row">
      <span className="score-label">{label}</span>
      <div className="bar" aria-hidden="true">
        {has && <div className={`bar-fill ${kind}`} style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />}
      </div>
      <span className="score-value">{has ? value.toFixed(2) : "none"}</span>
    </div>
  );
}

function MovieCard({ movie, rank }) {
  const overlap = (movie.overlapping_tags ?? []).map(tagName).filter(Boolean);
  const title = movie.movie?.title ?? movie.title ?? "Untitled";
  const rawYear = movie.movie?.release_date ? movie.movie.release_date.slice(0, 4) : null;
  const year = rawYear ?? movie.release_year ?? movie.year ?? movie.movie?.release_year;
  const posterPath = movie.movie?.poster_path ?? movie.poster_path;

  return (
    <li className="card">
      <MoviePoster path={posterPath} title={title} />
      <div className="card-body">
        <div className="card-head">
          <div className="card-title-group">
            <span className="rank">{rank}</span>
            <h3>{title}</h3>
            {year ? <span className="year">{year}</span> : null}
          </div>
          {typeof movie.final_score === "number" && (
            <span className="final">{movie.final_score.toFixed(2)}</span>
          )}
        </div>
        <div className="scores">
          <ScoreBar label="Tags" value={movie.tag_score} kind="tag" />
          <ScoreBar label="Plot" value={movie.plot_score} kind="plot" />
        </div>
        {movie.explanation && <p className="explanation">{movie.explanation}</p>}
        {overlap.length > 0 && (
          <details>
            <summary>Shared tags ({overlap.length})</summary>
            <p className="overlap">{overlap.join(", ")}</p>
          </details>
        )}
      </div>
    </li>
  );
}

function Skeleton() {
  return (
    <ul className="cards" aria-busy="true" aria-label="Loading recommendations">
      {Array.from({ length: 5 }).map((_, i) => (
        <li key={i} className="card skeleton">
          <div className="sk sk-poster" />
          <div className="card-body">
            <div className="sk sk-title" />
            <div className="sk sk-bar" />
            <div className="sk sk-bar short" />
            <div className="sk sk-text" />
          </div>
        </li>
      ))}
    </ul>
  );
}

export default function Results({ state, query, onClearFilters, onRetry }) {
  const { status, items, error } = state;

  if (status === "idle") {
    return (
      <div className="empty">
        <h2>Find something you have not seen</h2>
        <p>
          Pick a few movies you like, or search for a director, actor or keyword. Then choose how
          much plot and tags should count.
        </p>
      </div>
    );
  }

  if (status === "loading") return <Skeleton />;

  if (status === "error") {
    return (
      <div className="empty error-box" role="alert">
        <h2>Could not get recommendations</h2>
        <p>{error}</p>
        <button type="button" className="secondary" onClick={onRetry}>
          Try again
        </button>
      </div>
    );
  }

  if (items.length === 0) {
    return (
      <div className="empty">
        <h2>No movies match</h2>
        <p>
          {hasFilters(query)
            ? "Your filters left nothing to rank. Widen the date range or remove a language or genre."
            : "Nothing came back. Try different movies or tags."}
        </p>
        {hasFilters(query) && (
          <button type="button" className="secondary" onClick={onClearFilters}>
            Clear filters
          </button>
        )}
      </div>
    );
  }

  return (
    <>
      <h2 className="results-title">{items.length} recommendations</h2>
      <ul className="cards">
        {items.map((m, i) => (
          <MovieCard key={m.id ?? m.movie_id ?? i} movie={m} rank={i + 1} />
        ))}
      </ul>
    </>
  );
}
