// Dev: Vite proxies /api to localhost:8000. Prod: vercel.json rewrites /api to the EC2 backend.
const BASE = import.meta.env.VITE_API_URL || "/api";

async function request(path, options = {}) {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = "";
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* body was not JSON */
    }
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return res.json();
}

const asList = (data, ...keys) => {
  if (Array.isArray(data)) return data;
  for (const k of keys) if (Array.isArray(data?.[k])) return data[k];
  return [];
};

// Expected: [{ id, title, release_year, director }]
export async function searchMovies(q, signal) {
  return asList(await request(`/movies/search?q=${encodeURIComponent(q)}`, { signal }), "results");
}

// Expected: [{ tag_type, name }]
export async function searchTags(q, signal) {
  return asList(await request(`/tags/search?q=${encodeURIComponent(q)}`, { signal }), "results");
}

// Expected: { results: [{ id, title, release_year, final_score, tag_score, plot_score,
//                         explanation, overlapping_tags }] }
export async function recommend(body, signal) {
  const data = await request("/recommend", { method: "POST", body: JSON.stringify(body), signal });
  return asList(data, "results", "recommendations");
}
