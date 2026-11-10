function pageList(cur, total) {
  const keep = new Set([1, total, cur - 1, cur, cur + 1]);
  if (cur <= 3) [2, 3, 4].forEach((n) => keep.add(n));
  if (cur >= total - 2) [total - 3, total - 2, total - 1].forEach((n) => keep.add(n));
  const nums = [...keep].filter((n) => n >= 1 && n <= total).sort((a, b) => a - b);
  const out = [];
  nums.forEach((n, i) => {
    if (i > 0 && n - nums[i - 1] > 1) out.push("…");
    out.push(n);
  });
  return out;
}

export default function Pagination({ page, totalPages, onChange, disabled }) {
  if (totalPages <= 1) return null;
  return (
    <nav className="pagination" aria-label="Pagination">
      <button type="button" disabled={disabled || page === 1} onClick={() => onChange(page - 1)}>Previous</button>
      {pageList(page, totalPages).map((p, i) =>
        p === "…" ? (
          <span key={`gap${i}`} aria-hidden="true">…</span>
        ) : (
          <button type="button" key={p} disabled={disabled}
                  aria-current={p === page ? "page" : undefined}
                  className={p === page ? "active" : ""} onClick={() => onChange(p)}>{p}</button>
        )
      )}
      <button type="button" disabled={disabled || page === totalPages} onClick={() => onChange(page + 1)}>Next</button>
    </nav>
  );
}
