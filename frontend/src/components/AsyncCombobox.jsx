import { useEffect, useId, useRef, useState } from "react";
import { useDebounce } from "../hooks";

const MIN_CHARS = 2;
const MAX_VISIBLE = 10;

export default function AsyncCombobox({
  label,
  placeholder,
  fetchOptions,   // (query, signal) => Promise<item[]>
  onSelect,       // (item) => void
  getKey,         // (item) => string
  renderOption,   // (item) => ReactNode
  isSelected,     // (item) => boolean, selected items are hidden from suggestions
}) {
  const [text, setText] = useState("");
  const [options, setOptions] = useState([]);
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const [status, setStatus] = useState("idle"); // idle | loading | error
  const query = text.trim();
  const debounced = useDebounce(query, 250);
  const rootRef = useRef(null);
  const listId = useId();

  useEffect(() => {
    if (debounced.length < MIN_CHARS) {
      setOptions([]);
      setStatus("idle");
      return;
    }
    const controller = new AbortController();
    setStatus("loading");
    fetchOptions(debounced, controller.signal)
      .then((items) => {
        setOptions(items);
        setActive(0);
        setStatus("idle");
      })
      .catch((err) => {
        if (err.name !== "AbortError") setStatus("error");
      });
    return () => controller.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debounced]);

  useEffect(() => {
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const visible = options.filter((o) => !isSelected(o)).slice(0, MAX_VISIBLE);
  const settled = query.length >= MIN_CHARS && query === debounced && status !== "loading";

  function pick(item) {
    onSelect(item);
    setText("");
    setOptions([]);
    setOpen(false);
  }

  function onKeyDown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, Math.max(visible.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      if (open && visible[active]) {
        e.preventDefault();
        pick(visible[active]);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  const showPanel = open && query.length >= MIN_CHARS;

  return (
    <div className="combo" ref={rootRef}>
      <input
        type="text"
        role="combobox"
        aria-label={label}
        aria-expanded={showPanel}
        aria-controls={listId}
        aria-autocomplete="list"
        placeholder={placeholder}
        value={text}
        autoComplete="off"
        onChange={(e) => {
          setText(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={onKeyDown}
      />
      {showPanel && (
        <ul className="combo-list" id={listId} role="listbox">
          {status === "loading" && <li className="combo-note">Searching…</li>}
          {status === "error" && <li className="combo-note error">Search failed. Check that the backend is running.</li>}
          {settled && status === "idle" && visible.length === 0 && (
            <li className="combo-note">No matches</li>
          )}
          {visible.map((item, i) => (
            <li
              key={getKey(item)}
              role="option"
              aria-selected={i === active}
              className={i === active ? "combo-option active" : "combo-option"}
              onMouseEnter={() => setActive(i)}
              onMouseDown={(e) => e.preventDefault()}
              onClick={() => pick(item)}
            >
              {renderOption(item)}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
