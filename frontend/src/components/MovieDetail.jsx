import { useEffect, useState } from "react";
import { getMovie, isAbort } from "../api";
import { languageName } from "../constants";
import MoviePoster from "./MoviePoster";
import ScoreBar from "./ScoreBar";

function formatDate(dateStr) {
  if (!dateStr) return null;
  try {
    const [year, month, day] = dateStr.split("-").map(Number);
    if (!year || !month || !day) return dateStr;
    const date = new Date(year, month - 1, day);
    return date.toLocaleDateString("en-US", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  } catch {
    return dateStr;
  }
}

export default function MovieDetail({ movieId, initialMovie, onBack, onSeedMovie }) {
  const [fetchedMovie, setFetchedMovie] = useState(null);
  const [loading, setLoading] = useState(!initialMovie);
  const [error, setError] = useState("");

  const recommendation = initialMovie ?? null;
  const initialData = initialMovie?.movie ?? initialMovie ?? null;
  const movieData = fetchedMovie
    ? { ...(initialData || {}), ...fetchedMovie }
    : initialData;

  useEffect(() => {
    // Scroll to top when opening movie detail page
    window.scrollTo({ top: 0, behavior: "instant" });
  }, []);

  // Fetch full details from API (to populate or extend metadata)
  useEffect(() => {
    if (!movieId) return;
    const controller = new AbortController();

    async function load() {
      if (!initialMovie) setLoading(true);
      setError("");

      try {
        const full = await getMovie(movieId, controller.signal);
        setFetchedMovie(full);
      } catch (err) {
        if (isAbort(err)) return;
        if (!initialMovie) {
          setError(err.message || "Failed to load movie details.");
        }
      } finally {
        setLoading(false);
      }
    }

    load();
    return () => controller.abort();
  }, [movieId, initialMovie]);

  if (loading && !movieData) {
    return (
      <div className="movie-detail-page">
        <div className="detail-top-nav">
          <button type="button" className="btn-back" onClick={onBack}>
            <span aria-hidden="true">&larr;</span> Back to recommendations
          </button>
        </div>
        <div className="detail-card skeleton-detail">
          <div className="sk sk-detail-poster" />
          <div className="detail-main-content">
            <div className="sk sk-title" style={{ width: "60%", height: "32px" }} />
            <div className="sk sk-bar" style={{ width: "40%", height: "18px" }} />
            <div className="sk sk-text" style={{ width: "90%", height: "20px" }} />
            <div className="sk sk-text" style={{ width: "85%", height: "20px" }} />
            <div className="sk sk-text" style={{ width: "70%", height: "20px" }} />
          </div>
        </div>
      </div>
    );
  }

  if (error && !movieData) {
    return (
      <div className="movie-detail-page">
        <div className="detail-top-nav">
          <button type="button" className="btn-back" onClick={onBack}>
            <span aria-hidden="true">&larr;</span> Back to recommendations
          </button>
        </div>
        <div className="empty error-box" role="alert">
          <h2>Could not load movie details</h2>
          <p>{error}</p>
          <button type="button" className="secondary" onClick={onBack}>
            Return to recommendations
          </button>
        </div>
      </div>
    );
  }

  const title = movieData?.title ?? "Untitled Movie";
  const rawDate = movieData?.release_date;
  const year = rawDate ? rawDate.slice(0, 4) : movieData?.release_year ?? movieData?.year;
  const formattedDate = formatDate(rawDate);
  const posterPath = movieData?.poster_path;
  const overview = movieData?.overview;
  const genres = Array.isArray(movieData?.genres) ? movieData.genres : [];
  const directors = Array.isArray(movieData?.director) ? movieData.director : [];
  const cast = Array.isArray(movieData?.cast_members) ? movieData.cast_members : [];
  const languageCode = movieData?.original_language;
  const languageLabel = languageCode ? languageName(languageCode) : "";

  // Recommendation specific details
  const finalScore = typeof recommendation?.final_score === "number" ? recommendation.final_score : null;
  const tagScore = typeof recommendation?.tag_score === "number" ? recommendation.tag_score : null;
  const plotScore = typeof recommendation?.plot_score === "number" ? recommendation.plot_score : null;
  const explanation = recommendation?.explanation;
  const overlapTags = (recommendation?.overlapping_tags ?? []).map((t) => (typeof t === "string" ? t : t?.name)).filter(Boolean);

  // Extended metadata
  const writers = Array.isArray(movieData?.writers) ? movieData.writers : [];
  const producers = Array.isArray(movieData?.producers) ? movieData.producers : [];
  const composer = Array.isArray(movieData?.music_composer) ? movieData.music_composer : [];
  const companies = Array.isArray(movieData?.production_companies) ? movieData.production_companies : [];
  const countries = Array.isArray(movieData?.production_countries) ? movieData.production_countries : [];
  const spoken = Array.isArray(movieData?.spoken_languages) ? movieData.spoken_languages : [];

  return (
    <div className="movie-detail-page">
      <nav className="detail-top-nav" aria-label="Movie navigation">
        <button type="button" className="btn-back" onClick={onBack}>
          <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <line x1="19" y1="12" x2="5" y2="12" />
            <polyline points="12 19 5 12 12 5" />
          </svg>
          Back to recommendations
        </button>

        {onSeedMovie && (
          <button
            type="button"
            className="btn-seed"
            onClick={() => onSeedMovie({ id: movieData?.id ?? movieId, title })}
            title="Use this movie as an input seed in movie finder"
          >
            <svg className="nav-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="8" x2="12" y2="16" />
              <line x1="8" y1="12" x2="16" y2="12" />
            </svg>
            Find similar movies
          </button>
        )}
      </nav>

      <article className="detail-card">
        <div className="detail-poster-col">
          <MoviePoster path={posterPath} title={title} className="detail-poster" />
          {finalScore !== null && (
            <div className="detail-match-badge" title="Overall recommendation score">
              <span className="match-percent">{Math.round(finalScore * 100)}%</span>
              <span className="match-label">Match Score</span>
            </div>
          )}
        </div>

        <div className="detail-main-content">
          <header className="detail-header">
            <div className="detail-title-row">
              <h1 className="detail-title">{title}</h1>
              {year && <span className="detail-year">({year})</span>}
            </div>

            <div className="detail-meta-row">
              {formattedDate && (
                <span className="meta-item">
                  <strong>Released:</strong> {formattedDate}
                </span>
              )}
              {languageLabel && (
                <span className="meta-item">
                  <strong>Language:</strong> {languageLabel}
                  {languageCode && languageCode !== languageLabel.toLowerCase() && (
                    <span className="meta-code"> ({languageCode})</span>
                  )}
                </span>
              )}
            </div>

            {genres.length > 0 && (
              <div className="detail-genres" aria-label="Genres">
                {genres.map((g) => (
                  <span key={g} className="genre-pill">
                    {g}
                  </span>
                ))}
              </div>
            )}
          </header>

          {/* Overview (Plot) Section */}
          <section className="detail-section overview-section">
            <h2 className="section-heading">Overview</h2>
            {overview ? (
              <p className="detail-overview-text">{overview}</p>
            ) : (
              <p className="detail-overview-text text-muted">No plot summary available for this movie.</p>
            )}
          </section>

          {/* Recommendation Insights (if movie was recommended) */}
          {(finalScore !== null || explanation || overlapTags.length > 0) && (
            <section className="detail-section recommendation-box">
              <h2 className="section-heading">Why This Was Recommended</h2>
              {explanation && <p className="recommendation-explanation">{explanation}</p>}

              {(tagScore !== null || plotScore !== null) && (
                <div className="recommendation-scores-grid">
                  <div className="rec-score-item">
                    <span className="rec-score-header">Tags Match</span>
                    <ScoreBar label="Tags" value={tagScore} kind="tag" />
                  </div>
                  <div className="rec-score-item">
                    <span className="rec-score-header">Plot Similarity</span>
                    <ScoreBar label="Plot" value={plotScore} kind="plot" />
                  </div>
                </div>
              )}

              {overlapTags.length > 0 && (
                <div className="detail-overlap-block">
                  <h3 className="sub-heading">Shared Tags ({overlapTags.length})</h3>
                  <div className="chips">
                    {overlapTags.map((tag, i) => (
                      <span key={i} className="chip static">
                        {tag}
                      </span>
                    ))}
                  </div>
                </div>
              )}
            </section>
          )}

          {/* Directors */}
          {directors.length > 0 && (
            <section className="detail-section">
              <h2 className="section-heading">
                {directors.length === 1 ? "Director" : "Directors"}
              </h2>
              <div className="people-list">
                {directors.map((d, i) => (
                  <span key={i} className="person-badge">
                    {d}
                  </span>
                ))}
              </div>
            </section>
          )}

          {/* Cast Members */}
          {cast.length > 0 && (
            <section className="detail-section">
              <h2 className="section-heading">Cast Members ({cast.length})</h2>
              <div className="cast-grid">
                {cast.map((c, i) => (
                  <div key={i} className="cast-card">
                    <span className="cast-avatar" aria-hidden="true">
                      {c.slice(0, 1).toUpperCase()}
                    </span>
                    <span className="cast-name">{c}</span>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* Additional Crew & Production Info */}
          {(writers.length > 0 ||
            producers.length > 0 ||
            composer.length > 0 ||
            companies.length > 0 ||
            countries.length > 0 ||
            spoken.length > 0) && (
            <section className="detail-section crew-section">
              <h2 className="section-heading">Production &amp; Crew</h2>
              <dl className="crew-grid">
                {writers.length > 0 && (
                  <div className="crew-item">
                    <dt>Writers</dt>
                    <dd>{writers.join(", ")}</dd>
                  </div>
                )}
                {producers.length > 0 && (
                  <div className="crew-item">
                    <dt>Producers</dt>
                    <dd>{producers.join(", ")}</dd>
                  </div>
                )}
                {composer.length > 0 && (
                  <div className="crew-item">
                    <dt>Music Composer</dt>
                    <dd>{composer.join(", ")}</dd>
                  </div>
                )}
                {companies.length > 0 && (
                  <div className="crew-item">
                    <dt>Production Companies</dt>
                    <dd>{companies.join(", ")}</dd>
                  </div>
                )}
                {countries.length > 0 && (
                  <div className="crew-item">
                    <dt>Countries</dt>
                    <dd>{countries.join(", ")}</dd>
                  </div>
                )}
                {spoken.length > 0 && (
                  <div className="crew-item">
                    <dt>Spoken Languages</dt>
                    <dd>{spoken.join(", ")}</dd>
                  </div>
                )}
              </dl>
            </section>
          )}
        </div>
      </article>
    </div>
  );
}
