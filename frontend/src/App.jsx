import { useEffect, useRef, useState } from "react";
import QueryPanel from "./components/QueryPanel";
import Results from "./components/Results";
import MovieDetail from "./components/MovieDetail";
import { isAbort, recommend } from "./api";
import { INITIAL_QUERY, buildRequest } from "./query";

const STORAGE_PREFIX = "mf_movie_";
const RESULTS_KEY = "mf_last_results";

function getRouteFromLocation() {
  // Check hash first: #/movie/123 or #movie/123
  const hash = window.location.hash.replace(/^#\/?/, "");
  if (hash.startsWith("movie/")) {
    const id = hash.replace("movie/", "").trim();
    if (id) return { page: "movie", movieId: id };
  }

  // Check pathname: /movie/123
  const path = window.location.pathname;
  const match = path.match(/\/movie\/([^/?#]+)/);
  if (match) {
    return { page: "movie", movieId: match[1] };
  }

  return { page: "home", movieId: null };
}

function saveMovieToCache(id, data) {
  try {
    sessionStorage.setItem(`${STORAGE_PREFIX}${id}`, JSON.stringify(data));
  } catch {
    /* ignore storage errors */
  }
}

function getMovieFromCache(id) {
  try {
    const raw = sessionStorage.getItem(`${STORAGE_PREFIX}${id}`);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

function saveResultsToCache(items) {
  try {
    sessionStorage.setItem(RESULTS_KEY, JSON.stringify(items));
    for (const item of items) {
      const id = item.movie?.id ?? item.movie_id ?? item.id;
      if (id) saveMovieToCache(id, item);
    }
  } catch {
    /* ignore */
  }
}

function getResultsFromCache() {
  try {
    const raw = sessionStorage.getItem(RESULTS_KEY);
    return raw ? JSON.parse(raw) : null;
  } catch {
    return null;
  }
}

export default function App() {
  const [route, setRoute] = useState(() => getRouteFromLocation());
  const [selectedMovie, setSelectedMovie] = useState(() => {
    const r = getRouteFromLocation();
    return r.movieId ? getMovieFromCache(r.movieId) : null;
  });

  const [query, setQuery] = useState(INITIAL_QUERY);
  const [state, setState] = useState(() => {
    // If we have cached results in session, initialize with them
    const cached = getResultsFromCache();
    if (cached && cached.length > 0) {
      return { status: "done", items: cached, error: "" };
    }
    return { status: "idle", items: [], error: "" };
  });

  const abortRef = useRef(null);
  const resultsRef = useRef(null);

  // Listen for browser Back/Forward buttons and Hash changes
  useEffect(() => {
    function handleLocationChange() {
      const nextRoute = getRouteFromLocation();
      setRoute(nextRoute);

      if (nextRoute.page === "movie" && nextRoute.movieId) {
        const cached = getMovieFromCache(nextRoute.movieId);
        if (cached) setSelectedMovie(cached);
      } else {
        setSelectedMovie(null);
      }
    }

    window.addEventListener("popstate", handleLocationChange);
    window.addEventListener("hashchange", handleLocationChange);
    return () => {
      window.removeEventListener("popstate", handleLocationChange);
      window.removeEventListener("hashchange", handleLocationChange);
    };
  }, []);

  function navigateToMovie(movieItem) {
    const id = movieItem.movie?.id ?? movieItem.movie_id ?? movieItem.id;
    if (!id) return;

    saveMovieToCache(id, movieItem);
    setSelectedMovie(movieItem);
    setRoute({ page: "movie", movieId: String(id) });

    const newUrl = `/movie/${id}`;
    if (window.location.pathname !== newUrl) {
      window.history.pushState({ page: "movie", movieId: id }, "", newUrl);
    }
  }

  function navigateToHome() {
    setSelectedMovie(null);
    setRoute({ page: "home", movieId: null });

    if (window.location.pathname !== "/" || window.location.hash) {
      window.history.pushState({ page: "home" }, "", "/");
    }
  }

  function handleSeedMovie(movie) {
    if (!movie?.id) return;
    setQuery((q) => {
      if (q.movies.some((m) => String(m.id) === String(movie.id))) {
        return q;
      }
      return {
        ...q,
        movies: [...q.movies, { id: movie.id, title: movie.title || "Movie" }],
      };
    });
    navigateToHome();
  }

  async function submit() {
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;
    setState({ status: "loading", items: [], error: "" });

    if (window.matchMedia("(max-width: 900px)").matches) {
      resultsRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
    }

    try {
      const items = await recommend(buildRequest(query), controller.signal);
      setState({ status: "done", items, error: "" });
      saveResultsToCache(items);
    } catch (err) {
      if (isAbort(err)) return;
      setState({ status: "error", items: [], error: err.message || "Something went wrong." });
    }
  }

  function clearFilters() {
    setQuery((q) => ({ ...q, languages: [], genres: [], dateFrom: "", dateTo: "" }));
    setState({ status: "idle", items: [], error: "" });
    try {
      sessionStorage.removeItem(RESULTS_KEY);
    } catch {
      /* ignore */
    }
  }

  return (
    <div className="app">
      <header className="site-header">
        <h1>
          <a
            href="/"
            className="brand-link"
            onClick={(e) => {
              e.preventDefault();
              navigateToHome();
            }}
          >
            Movie finder
          </a>
        </h1>
        <p>Content-based picks.</p>
      </header>

      {route.page === "movie" ? (
        <main>
          <MovieDetail
            key={route.movieId}
            movieId={route.movieId}
            initialMovie={selectedMovie}
            onBack={navigateToHome}
            onSeedMovie={handleSeedMovie}
          />
        </main>
      ) : (
        <main className="layout">
          <QueryPanel
            query={query}
            setQuery={setQuery}
            onSubmit={submit}
            loading={state.status === "loading"}
          />
          <section className="results" ref={resultsRef} aria-live="polite">
            <Results
              state={state}
              query={query}
              onClearFilters={clearFilters}
              onRetry={submit}
              onSelectMovie={navigateToMovie}
            />
          </section>
        </main>
      )}
    </div>
  );
}
