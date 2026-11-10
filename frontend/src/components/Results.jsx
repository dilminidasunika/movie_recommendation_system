import { hasFilters } from "../query";
import MoviePoster from "./MoviePoster";
import ScoreBar from "./ScoreBar";
import Pagination from "./Pagination";

const tagName = (t) => (typeof t === "string" ? t : t?.name ?? "");

function MovieCard({ movie, rank, explaining, onSelect }) {
  const overlap = (movie.overlapping_tags ?? []).map(tagName).filter(Boolean);
  const title = movie.movie?.title ?? movie.title ?? "Untitled";
  const rawYear = movie.movie?.release_date ? movie.movie.release_date.slice(0, 4) : null;
  const year = rawYear ?? movie.release_year ?? movie.year ?? movie.movie?.release_year;
  const posterPath = movie.movie?.poster_path ?? movie.poster_path;

  function handleClick(e) {
    if (e.target.closest("details") || e.target.closest("button")) {
      return;
    }
    onSelect?.(movie);
  }

  function handleKeyDown(e) {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      onSelect?.(movie);
    }
  }

  return (
    <li
      className="card card-clickable"
      tabIndex={0}
      role="button"
      onClick={handleClick}
      onKeyDown={handleKeyDown}
      aria-label={`View details for ${title}`}
    >
      <MoviePoster path={posterPath} title={title} />
      <div className="card-body">
        <div className="card-head">
          <div className="card-title-group">
            <span className="rank">{rank}</span>
            <h3 className="card-title-link">{title}</h3>
            {year ? <span className="year">({year})</span> : null}
          </div>
          {typeof movie.final_score === "number" && (
            <span className="final">{movie.final_score.toFixed(2)}</span>
          )}
        </div>
        <div className="scores">
          <ScoreBar label="Tags" value={movie.tag_score} kind="tag" />
          <ScoreBar label="Plot" value={movie.plot_score} kind="plot" />
        </div>
        {movie.explanation
          ? <p className="explanation">{movie.explanation}</p>
          : explaining ? <div className="sk sk-text" /> : null}
        {overlap.length > 0 && (
          <details onClick={(e) => e.stopPropagation()}>
            <summary>Shared tags ({overlap.length})</summary>
            <p className="overlap">{overlap.join(", ")}</p>
          </details>
        )}
        <div className="card-action-bar">
          <span className="card-action-text">
            View details &amp; plot <span aria-hidden="true">&rarr;</span>
          </span>
        </div>
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

export default function Results({
  state,
  query,
  onClearFilters,
  onRetry,
  onSelectMovie,
  onPageChange,
}) {
  const { status, error, pages, page, total, pageLoading, explaining } = state;
  const items = pages?.[page] ?? [];
  const totalPages = Math.ceil((total || 0) / 20);

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

  if (total === 0) {
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
      <h2 className="results-title">{total} recommendations · page {page} of {totalPages}</h2>
      {pageLoading ? <Skeleton /> : (
        <ul className="cards">
          {items.map((m, i) => (
            <MovieCard
              key={m.movie_id}
              movie={m}
              rank={(page - 1) * 20 + i + 1}
              explaining={explaining?.[page]}
              onSelect={onSelectMovie}
            />
          ))}
        </ul>
      )}
      <Pagination page={page} totalPages={totalPages} onChange={onPageChange} disabled={pageLoading} />
    </>
  );
}
