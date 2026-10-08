// Values are lowercase on purpose: the backend lowercases and the DB stores lowercase.

export const LANGUAGES = [
  { value: "en", label: "English" },
  { value: "fr", label: "French" },
  { value: "es", label: "Spanish" },
  { value: "de", label: "German" },
  { value: "it", label: "Italian" },
  { value: "pt", label: "Portuguese" },
  { value: "ru", label: "Russian" },
  { value: "ja", label: "Japanese" },
  { value: "ko", label: "Korean" },
  { value: "zh", label: "Chinese" },
  { value: "hi", label: "Hindi" },
  { value: "ta", label: "Tamil" },
  { value: "si", label: "Sinhala" },
];

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
