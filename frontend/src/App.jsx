import { useRef, useState } from "react";
import QueryPanel from "./components/QueryPanel";
import Results from "./components/Results";
import { recommend } from "./api";
import { INITIAL_QUERY, buildRequest } from "./query";

export default function App() {
  const [query, setQuery] = useState(INITIAL_QUERY);
  const [state, setState] = useState({ status: "idle", items: [], error: "" });
  const abortRef = useRef(null);
  const resultsRef = useRef(null);

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
    } catch (err) {
      if (err.name === "AbortError") return;
      setState({ status: "error", items: [], error: err.message });
    }
  }

  function clearFilters() {
    setQuery((q) => ({ ...q, languages: [], genres: [], dateFrom: "", dateTo: "" }));
    setState({ status: "idle", items: [], error: "" });
  }

  return (
    <div className="app">
      <header className="site-header">
        <h1>Movie finder</h1>
        <p>Content-based picks, with a reason for each one.</p>
      </header>
      <main className="layout">
        <QueryPanel
          query={query}
          setQuery={setQuery}
          onSubmit={submit}
          loading={state.status === "loading"}
        />
        <section className="results" ref={resultsRef} aria-live="polite">
          <Results state={state} query={query} onClearFilters={clearFilters} />
        </section>
      </main>
    </div>
  );
}
