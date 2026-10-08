import AsyncCombobox from "./AsyncCombobox";
import ChipList from "./ChipList";
import { searchMovies, searchTags } from "../api";
import { GENRES, LANGUAGES, TAG_LABELS } from "../constants";
import { canSubmit, dateRangeInvalid } from "../query";

const tagKey = (t) => `${t.tag_type}:${t.name}`;
const tagLabel = (t) => `${TAG_LABELS[t.tag_type] ?? t.tag_type}: ${t.name}`;
const movieYear = (m) => m.release_year ?? m.year ?? "";

function toggle(list, value) {
  return list.includes(value) ? list.filter((v) => v !== value) : [...list, value];
}

function ToggleGroup({ legend, options, selected, onToggle }) {
  return (
    <fieldset className="field">
      <legend>{legend}</legend>
      <div className="toggles">
        {options.map((o) => {
          const value = typeof o === "string" ? o : o.value;
          const label = typeof o === "string" ? o : o.label;
          const on = selected.includes(value);
          return (
            <button
              key={value}
              type="button"
              className={on ? "toggle on" : "toggle"}
              aria-pressed={on}
              onClick={() => onToggle(value)}
            >
              {label}
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}

export default function QueryPanel({ query, setQuery, onSubmit, loading }) {
  const set = (patch) => setQuery((q) => ({ ...q, ...patch }));
  const noMovies = query.movies.length === 0;
  const tagPct = Math.round(query.tagWeight * 100);
  const badRange = dateRangeInvalid(query);

  return (
    <form
      className="panel"
      onSubmit={(e) => {
        e.preventDefault();
        if (canSubmit(query) && !loading) onSubmit();
      }}
    >
      <div className="field">
        <label>Movies you like</label>
        <AsyncCombobox
          label="Search movies"
          placeholder="Search by title"
          fetchOptions={searchMovies}
          onSelect={(m) => set({ movies: [...query.movies, m] })}
          getKey={(m) => String(m.id)}
          isSelected={(m) => query.movies.some((x) => x.id === m.id)}
          renderOption={(m) => (
            <>
              <span className="opt-title">{m.title}</span>
              <span className="opt-meta">
                {[movieYear(m), m.director].filter(Boolean).join(", ")}
              </span>
            </>
          )}
        />
        <ChipList
          items={query.movies}
          getKey={(m) => String(m.id)}
          getLabel={(m) => (movieYear(m) ? `${m.title} (${movieYear(m)})` : m.title)}
          onRemove={(m) => set({ movies: query.movies.filter((x) => x.id !== m.id) })}
        />
      </div>

      <div className="field">
        <label>People, keywords and more</label>
        <AsyncCombobox
          label="Search tags"
          placeholder="Director, cast, keyword, country"
          fetchOptions={searchTags}
          onSelect={(t) => set({ tags: [...query.tags, t] })}
          getKey={tagKey}
          isSelected={(t) => query.tags.some((x) => tagKey(x) === tagKey(t))}
          renderOption={(t) => (
            <>
              <span className="opt-title">{t.name}</span>
              <span className="opt-badge">{TAG_LABELS[t.tag_type] ?? t.tag_type}</span>
            </>
          )}
        />
        <ChipList
          items={query.tags}
          getKey={tagKey}
          getLabel={tagLabel}
          onRemove={(t) => set({ tags: query.tags.filter((x) => tagKey(x) !== tagKey(t)) })}
        />
      </div>

      <div className="divider" />

      <p className="note">
        Language, genre and date are hard filters. Tags only nudge similarity, so they do not
        guarantee a person's films appear.
      </p>

      <ToggleGroup
        legend="Original language"
        options={LANGUAGES}
        selected={query.languages}
        onToggle={(v) => set({ languages: toggle(query.languages, v) })}
      />

      <ToggleGroup
        legend="Genres (any of)"
        options={GENRES}
        selected={query.genres}
        onToggle={(v) => set({ genres: toggle(query.genres, v) })}
      />

      <fieldset className="field">
        <legend>Release date</legend>
        <div className="dates">
          <label>
            <span>From</span>
            <input type="date" value={query.dateFrom} onChange={(e) => set({ dateFrom: e.target.value })} />
          </label>
          <label>
            <span>To</span>
            <input type="date" value={query.dateTo} onChange={(e) => set({ dateTo: e.target.value })} />
          </label>
        </div>
        {badRange ? (
          <p className="note error">The start date must be on or before the end date.</p>
        ) : (
          <p className="note">Movies with no release date are left out when you set a date.</p>
        )}
      </fieldset>

      <div className="divider" />

      <div className="field">
        <label htmlFor="weight">Match on</label>
        <input
          id="weight"
          type="range"
          min="0"
          max="1"
          step="0.05"
          value={noMovies ? 1 : query.tagWeight}
          disabled={noMovies}
          onChange={(e) => set({ tagWeight: Number(e.target.value) })}
        />
        <div className="slider-labels">
          <span>Plot</span>
          <span>Tags</span>
        </div>
        <p className="note">
          {noMovies
            ? "Tags only. Pick at least one movie to compare plots."
            : `${tagPct}% tags, ${100 - tagPct}% plot`}
        </p>
      </div>

      <button className="primary" type="submit" disabled={!canSubmit(query) || loading}>
        {loading ? "Finding movies…" : "Find movies"}
      </button>
      {!canSubmit(query) && !badRange && (
        <p className="note">Add at least one movie or tag to start.</p>
      )}
    </form>
  );
}
