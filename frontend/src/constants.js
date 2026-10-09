// Values are lowercase on purpose: the backend lowercases and the DB stores lowercase.

// ISO 639-1 codes (plus "cn" and "xx" as used by TMDB-style data). Trim to what your DB contains.
const LANGUAGE_CODES =
  "en fr es de it pt ru ja ko zh cn hi ta te ml kn bn mr gu pa ur si ne ar fa he tr el pl cs sk hu ro bg sr hr sl uk be lt lv et fi sv no nb da nl is ga cy ca eu gl sq mk bs mt hy ka az kk uz mn th vi id ms tl km lo my sw af zu xh am so ha yo ig la eo sa yi xx".split(" ");

const SPECIAL = { cn: "Cantonese", xx: "No language" };
let names = null;
try {
  names = new Intl.DisplayNames(["en"], { type: "language" });
} catch {
  /* older browser: fall back to codes */
}

function languageName(code) {
  if (SPECIAL[code]) return SPECIAL[code];
  try {
    return names?.of(code) ?? code;
  } catch {
    return code;
  }
}

export const LANGUAGES = LANGUAGE_CODES.map((value) => ({
  value,
  label: `${languageName(value)} (${value})`,
})).sort((a, b) => a.label.localeCompare(b.label));

export const GENRES = [
  "action", "adventure", "animation", "comedy", "crime", "documentary",
  "drama", "family", "fantasy", "history", "horror", "music", "mystery",
  "romance", "science fiction", "thriller", "war", "western",
];

// Must match the tag_lookup field names used by the backend exactly.
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
