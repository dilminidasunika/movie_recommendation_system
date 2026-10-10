// Values are lowercase on purpose: the backend lowercases and the DB stores lowercase.

export const GENRES = [
  "action", "adventure", "animation", "comedy", "crime", "documentary",
  "drama", "family", "fantasy", "history", "horror", "music", "mystery",
  "romance", "science fiction", "thriller", "war", "western",
];

// Must match the tag_lookup tag_type names used by the backend exactly.
// original_language is NOT here: it is a hard filter with its own dropdown.
export const TAG_LABELS = {
  cast_members: "Cast",
  director: "Director",
  writers: "Writer",
  producers: "Producer",
  music_composer: "Composer",
  keywords: "Keyword",
  production_companies: "Company",
  production_countries: "Country",
  spoken_languages: "Spoken language",
};

export const TAG_TYPES = Object.keys(TAG_LABELS);

// ISO 639-1 code -> English name, using the browser's built-in Intl.DisplayNames.
const SPECIAL = { cn: "Cantonese", xx: "No language", sh: "Serbo-Croatian" };

let displayNames = null;
try {
  displayNames = new Intl.DisplayNames(["en"], { type: "language" });
} catch {
  /* very old browser: falls back to the code */
}

export function languageName(code) {
  const c = String(code).toLowerCase();
  if (SPECIAL[c]) return SPECIAL[c];
  try {
    const name = displayNames?.of(c);
    return name && name !== c ? name : c; // of() returns the code itself when unknown
  } catch {
    return c;
  }
}

// Used only to let people type a language NAME ("tamil") and find its code ("ta").
// The backend searches codes, so this list just translates names to codes.
const LANGUAGE_CODES =
  "en fr es de it pt ru ja ko zh cn hi ta te ml kn bn mr gu pa ur si ne ar fa he tr el pl cs sk hu ro bg sr hr sl uk be lt lv et fi sv no nb da nl is ga cy ca eu gl sq mk bs mt hy ka az kk uz mn th vi id ms tl km lo my sw af zu xh am so ha yo ig la eo sa yi xx".split(" ");

export function codesMatchingName(text) {
  const t = text.trim().toLowerCase();
  if (t.length < 2) return [];
  return LANGUAGE_CODES.filter((c) => languageName(c).toLowerCase().includes(t));
}
