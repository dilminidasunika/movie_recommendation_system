export default function ScoreBar({ label, value, kind }) {
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
