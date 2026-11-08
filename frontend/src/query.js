export const INITIAL_QUERY = {
  movies: [],      // [{ id, title, ... }]
  tags: [],        // [{ tag_type, name }]
  plotText: "",    // optional plot description
  languages: [],   // ["en", "fr"]
  genres: [],      // ["action"]
  dateFrom: "",
  dateTo: "",
  tagWeight: 0.5,  // 0 = plot only, 1 = tags only
};

export const TOP_N = 20;

export function dateRangeInvalid(q) {
  return Boolean(q.dateFrom && q.dateTo && q.dateFrom > q.dateTo);
}

export function getPlotTextError(text) {
  if (!text) return "";
  const raw = String(text);
  const collapsed = raw.trim().replace(/\s+/g, " ");
  if (collapsed.length === 0) return "";
  if (collapsed.length < 30) {
    return `Plot description must be at least 30 characters (${collapsed.length}/30).`;
  }
  if (raw.length > 1000 || collapsed.length > 1000) {
    return `Plot description must not exceed 1000 characters (${raw.length}/1000).`;
  }
  return "";
}

export function hasPlotQuery(q) {
  const hasMovies = Boolean(q.movies && q.movies.length > 0);
  const collapsedPlot = (q.plotText || "").trim().replace(/\s+/g, " ");
  const hasPlotText = collapsedPlot.length >= 30 && (q.plotText || "").length <= 1000;
  return hasMovies || hasPlotText;
}

// Backend rule: require at least one of movie_ids, tags, or plot_text.
export function canSubmit(q) {
  if (dateRangeInvalid(q)) return false;
  if (getPlotTextError(q.plotText)) return false;
  const hasMovies = Boolean(q.movies && q.movies.length > 0);
  const hasTags = Boolean(q.tags && q.tags.length > 0);
  const collapsedPlot = (q.plotText || "").trim().replace(/\s+/g, " ");
  const hasValidPlotText = collapsedPlot.length >= 30 && (q.plotText || "").length <= 1000;
  return hasMovies || hasTags || hasValidPlotText;
}

export function hasFilters(q) {
  return Boolean(q.languages?.length || q.genres?.length || q.dateFrom || q.dateTo);
}

// Empty arrays and empty dates are omitted. If there are no movies and no plot
// text, there is no plot query, so the weight is forced to 1.0 (the server does the same).
export function buildRequest(q) {
  const hasMovies = Boolean(q.movies && q.movies.length > 0);
  const collapsedPlot = (q.plotText || "").trim().replace(/\s+/g, " ");
  const hasPlot = hasMovies || Boolean(collapsedPlot);

  const req = { tag_weight: hasPlot ? q.tagWeight : 1.0, top_n: TOP_N };
  if (hasMovies) req.movie_ids = q.movies.map((m) => m.id);
  if (collapsedPlot) req.plot_text = collapsedPlot;

  const tags = (q.tags || []).map((t) => ({ tag_type: t.tag_type, name: t.name }));

  if (q.genres && q.genres.length) {
    req.genres = q.genres;
    for (const g of q.genres) {
      if (!tags.some((t) => t.tag_type === "genres" && t.name === g)) {
        tags.push({ tag_type: "genres", name: g });
      }
    }
  }
  if (tags.length) req.tags = tags;

  if (q.languages && q.languages.length) req.original_language = q.languages;
  if (q.dateFrom) req.date_from = q.dateFrom;
  if (q.dateTo) req.date_to = q.dateTo;
  return req;
}
