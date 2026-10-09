import { useEffect, useId, useMemo, useRef, useState } from "react";
import ChipList from "./ChipList";

// Searchable dropdown with multiple selection. Selected items show as chips underneath.
export default function MultiSelect({ label, placeholder, options, selected, onChange }) {
  const [text, setText] = useState("");
  const [open, setOpen] = useState(false);
  const [active, setActive] = useState(0);
  const rootRef = useRef(null);
  const listId = useId();

  const filtered = useMemo(() => {
    const q = text.trim().toLowerCase();
    if (!q) return options;
    return options.filter((o) => o.label.toLowerCase().includes(q) || o.value.toLowerCase().includes(q));
  }, [options, text]);

  useEffect(() => {
    const onDown = (e) => {
      if (rootRef.current && !rootRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const toggle = (value) =>
    onChange(selected.includes(value) ? selected.filter((v) => v !== value) : [...selected, value]);

  function onKeyDown(e) {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setOpen(true);
      setActive((i) => Math.min(i + 1, Math.max(filtered.length - 1, 0)));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setActive((i) => Math.max(i - 1, 0));
    } else if (e.key === "Enter") {
      if (open && filtered[active]) {
        e.preventDefault();
        toggle(filtered[active].value);
      }
    } else if (e.key === "Escape") {
      setOpen(false);
    }
  }

  const selectedOptions = selected.map((v) => options.find((o) => o.value === v) ?? { value: v, label: v });

  return (
    <div className="multi">
      <div className="combo" ref={rootRef}>
        <input
          type="text"
          role="combobox"
          aria-label={label}
          aria-expanded={open}
          aria-controls={listId}
          placeholder={selected.length ? `${selected.length} selected, search to add more` : placeholder}
          value={text}
          autoComplete="off"
          onChange={(e) => {
            setText(e.target.value);
            setActive(0);
            setOpen(true);
          }}
          onFocus={() => setOpen(true)}
          onKeyDown={onKeyDown}
        />
        {open && (
          <ul className="combo-list" id={listId} role="listbox" aria-multiselectable="true">
            {filtered.length === 0 && <li className="combo-note">No matches</li>}
            {filtered.map((o, i) => {
              const on = selected.includes(o.value);
              return (
                <li
                  key={o.value}
                  role="option"
                  aria-selected={on}
                  className={i === active ? "combo-option active" : "combo-option"}
                  onMouseEnter={() => setActive(i)}
                  onMouseDown={(e) => e.preventDefault()}
                  onClick={() => toggle(o.value)}
                >
                  <span className="opt-title">{o.label}</span>
                  <span className={on ? "tick on" : "tick"} aria-hidden="true">{on ? "✓" : ""}</span>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <ChipList
        items={selectedOptions}
        getKey={(o) => o.value}
        getLabel={(o) => o.label}
        onRemove={(o) => toggle(o.value)}
      />
    </div>
  );
}
