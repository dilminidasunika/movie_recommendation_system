import axios from "axios";
import { TAG_TYPES, codesMatchingName } from "./constants";

// ---------------------------------------------------------------------------
// Client. Vite proxies /api to http://localhost:8000 (see vite.config.js).
// ---------------------------------------------------------------------------
const http = axios.create({
  baseURL: "http://localhost:8000/api",
  timeout: 10_000, // searches; /recommend overrides this because of the LLM call
  headers: { "Content-Type": "application/json" },
});

// ---------------------------------------------------------------------------
// Errors: every failure becomes an ApiError with a readable message and a kind,
// except cancellations, which are re-thrown untouched (use isAbort to detect them).
// kind: network | timeout | unavailable | server | client | not_found | rate_limit | bad_response
// ---------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(message, { kind = "server", status } = {}) {
    super(message);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
  }
}

export const isAbort = (err) => axios.isCancel(err) || err?.name === "AbortError";

// FastAPI 422 -> "tags.0.tag_type: Input should be 'director', ..."; string detail -> as is.
function readDetail(data) {
  const d = data?.detail;
  if (typeof d === "string") return d;
  if (Array.isArray(d)) {
    return d
      .map((e) => {
        const where = Array.isArray(e.loc) ? e.loc.filter((p) => p !== "body").join(".") : "";
        return where ? `${where}: ${e.msg}` : e.msg;
      })
      .join("; ");
  }
  return "";
}

function toApiError(err) {
  if (err instanceof ApiError) return err;
  if (err.code === "ECONNABORTED" || err.code === "ETIMEDOUT") {
    return new ApiError("The request timed out. Try again.", { kind: "timeout" });
  }
  if (!err.response) {
    return new ApiError("Cannot reach the server. Is the backend running on port 8000?", { kind: "network" });
  }
  const { status, data } = err.response;
  const detail = readDetail(data);
  if (status === 404) return new ApiError(detail || "Endpoint not found. Check the backend routes.", { kind: "not_found", status });
  if (status === 429) return new ApiError("Too many requests. Wait a moment and try again.", { kind: "rate_limit", status });
  if (status === 400 || status === 422) return new ApiError(detail ? `Invalid request: ${detail}` : "Invalid request.", { kind: "client", status });
  if (status === 502 || status === 503 || status === 504) {
    return new ApiError("The server is not responding. Is the backend running?", { kind: "unavailable", status });
  }
  return new ApiError(detail || `Server error (${status}).`, { kind: "server", status });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// One retry for transient failures, only where it is cheap and safe (searches).
async function call(config, { retry = false } = {}) {
  for (let attempt = 0; ; attempt++) {
    try {
      return (await http.request(config)).data;
    } catch (err) {
      if (axios.isCancel(err)) throw err;
      const apiErr = toApiError(err);
      const transient = apiErr.kind === "network" || apiErr.kind === "unavailable";
      if (retry && transient && attempt === 0) {
        await sleep(300);
        continue; // if the signal was aborted meanwhile, the next request rejects as a cancel
      }
      throw apiErr;
    }
  }
}

function list(data, ...keys) {
  if (Array.isArray(data)) return data;
  for (const k of keys) if (Array.isArray(data?.[k])) return data[k];
  throw new ApiError("Unexpected response from the server.", { kind: "bad_response" });
}

// ---------------------------------------------------------------------------
// Session-only cache for searches (memory only, nothing persisted). Failures are never cached.
// ---------------------------------------------------------------------------
const TTL_MS = 5 * 60_000;
const MAX_ENTRIES = 300;
const cache = new Map();

async function cached(key, load) {
  const hit = cache.get(key);
  if (hit && hit.exp > Date.now()) {
    cache.delete(key); // re-insert so the most recently used entry stays newest
    cache.set(key, hit);
    return hit.value;
  }
  const value = await load();
  cache.set(key, { value, exp: Date.now() + TTL_MS });
  if (cache.size > MAX_ENTRIES) cache.delete(cache.keys().next().value);
  return value;
}

// ---------------------------------------------------------------------------
// Endpoints
// ---------------------------------------------------------------------------

// GET /api/movies/search?q=  ->  [{ id, title, release_year, director }]
export function searchMovies(q, signal) {
  const params = { q: q.trim() };
  return cached(`movies:${params.q}`, async () =>
    list(await call({ url: "/movies/search", params, signal }, { retry: true }), "results")
  );
}

// GET /api/tags/search?q=&types=a,b&per_type=5  ->  [{ tag_type, name }]
// The tag box always sends an explicit types list WITHOUT original_language, so a language
// can never become a soft tag.
export function searchTags(q, signal, { types = TAG_TYPES, perType = 5 } = {}) {
  const params = { q: q.trim(), types: types.join(","), per_type: perType };
  return cached(`tags:${params.types}:${perType}:${params.q}`, async () =>
    list(await call({ url: "/tags/search", params, signal }, { retry: true }), "results")
  );
}

// Same endpoint and behaviour as the tag box, limited to original_language.
// Typing a code ("ta") goes straight to the backend. Typing a name ("tamil") is translated to
// codes locally, each code is searched too, and the results are merged. One failed extra
// lookup does not break the others.
export async function searchLanguages(q, signal) {
  const ql = q.trim().toLowerCase();
  const queries = [...new Set([ql, ...codesMatchingName(ql).slice(0, 3)])];
  const settled = await Promise.allSettled(
    queries.map((x) => searchTags(x, signal, { types: ["original_language"], perType: 10 }))
  );
  const ok = settled.filter((r) => r.status === "fulfilled").map((r) => r.value);
  if (ok.length === 0) throw settled[0].reason;

  const seen = new Set();
  const out = [];
  for (const rows of ok) {
    for (const r of rows) {
      const code = String(r.name).toLowerCase();
      if (!seen.has(code)) {
        seen.add(code);
        out.push({ tag_type: "original_language", name: code });
      }
    }
  }
  return out;
}

// POST /api/recommend  ->  { results: [...] }
// Long timeout (LLM explanations), no retry (expensive), never cached.
export async function recommend(body, signal) {
  console.log("Calling /recommend with body:", body);
  const data = await call({ url: "/recommend", method: "POST", data: body, signal, timeout: 120_000 });
  console.log("Response from /recommend:", data);
  return list(data, "results", "recommendations");
}
