import { hasFilters } from "../query";

const tagName = (t) => (typeof t === "string" ? t : t?.name ?? "");

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
  const year = movie.release_year ?? movie.year;
  return (
    <li className="card">
      <div className="card-head">
        <span className="rank">{rank}</span>
        <h3>{movie.title}</h3>
        {year ? <span className="year">{year}</span> : null}
        <span className="final">{typeof movie.final_score === "number" ? movie.final_score.toFixed(2) : ""}</span>
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
    </li>
  );
}

function Skeleton() {
  return (
    <ul className="cards" aria-busy="true" aria-label="Loading recommendations">
      {Array.from({ length: 5 }).map((_, i) => (
        <li key={i} className="card skeleton">
          <div className="sk sk-title" />
          <div className="sk sk-bar" />
          <div className="sk sk-bar short" />
          <div className="sk sk-text" />
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
          <MovieCard key={m.id ?? i} movie={m} rank={i + 1} />
        ))}
      </ul>
    </>
  );
}
