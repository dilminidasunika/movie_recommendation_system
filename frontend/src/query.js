export const INITIAL_QUERY = {
  movies: [],      // [{ id, title, ... }]
  tags: [],        // [{ tag_type, name }]
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

// Backend rule: at least one movie or one tag.
export function canSubmit(q) {
  return (q.movies.length > 0 || q.tags.length > 0) && !dateRangeInvalid(q);
}

export function hasFilters(q) {
  return Boolean(q.languages.length || q.genres.length || q.dateFrom || q.dateTo);
}

// Empty arrays and empty dates are omitted. With no sample movies there is no plot
// centroid, so the weight is forced to 1.0 (the server does the same).
export function buildRequest(q) {
  const hasMovies = q.movies.length > 0;
  const req = { tag_weight: hasMovies ? q.tagWeight : 1.0, top_n: TOP_N };
  if (hasMovies) req.movie_ids = q.movies.map((m) => m.id);
  if (q.tags.length) req.tags = q.tags.map((t) => ({ tag_type: t.tag_type, name: t.name }));
  if (q.languages.length) req.original_language = q.languages;
  if (q.genres.length) req.genres = q.genres;
  if (q.dateFrom) req.date_from = q.dateFrom;
  if (q.dateTo) req.date_to = q.dateTo;
  return req;
}
