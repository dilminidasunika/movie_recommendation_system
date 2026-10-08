export default function ChipList({ items, getKey, getLabel, onRemove }) {
  if (items.length === 0) return null;
  return (
    <ul className="chips">
      {items.map((item) => (
        <li key={getKey(item)} className="chip">
          <span>{getLabel(item)}</span>
          <button type="button" aria-label={`Remove ${getLabel(item)}`} onClick={() => onRemove(item)}>
            ×
          </button>
        </li>
      ))}
    </ul>
  );
}
