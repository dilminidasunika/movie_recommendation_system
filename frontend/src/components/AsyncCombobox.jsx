import { useEffect, useId, useRef, useState } from "react";
import { useDebounce } from "../hooks";
import { isAbort } from "../api";

const MIN_CHARS = 2;
const MAX_VISIBLE = 45;

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
  const [errorMsg, setErrorMsg] = useState("");
  const query = text.trim();
  const debounced = useDebounce(query, 250);
  const rootRef = useRef(null);
  const listId = useId();

  const isShort = debounced.length < MIN_CHARS;
  const currentOptions = isShort ? [] : options;
  const currentStatus = isShort ? "idle" : status;

  useEffect(() => {
    if (debounced.length < MIN_CHARS) {
      return;
    }
    const controller = new AbortController();
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setStatus("loading");
    fetchOptions(debounced, controller.signal)
      .then((items) => {
        setOptions(items);
        setActive(0);
        setStatus("idle");
      })
      .catch((err) => {
        if (isAbort(err)) return;
        setOptions([]);
        setErrorMsg(err.message || "Search failed.");
        setStatus("error");
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

  const visible = currentOptions.filter((o) => !isSelected(o)).slice(0, MAX_VISIBLE);
  const settled = debounced === query && debounced.length >= MIN_CHARS;

  function pick(item) {
    onSelect(item);
    setText("");
    setOptions([]);
    setOpen(false);
  }

  function onKeyDown(e) {
    if (!open && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      setOpen(true);
      return;
    }
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setActive((a) => (visible.length ? (a + 1) % visible.length : 0));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((a) => (visible.length ? (a - 1 + visible.length) % visible.length : 0));
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
          {currentStatus === "loading" && <li className="combo-note">Searching…</li>}
          {currentStatus === "error" && <li className="combo-note error">{errorMsg}</li>}
          {settled && currentStatus === "idle" && visible.length === 0 && (
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
