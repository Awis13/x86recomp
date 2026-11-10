#pragma once
// The user's language and the locales the system has, as kernel32 tells
// the game (GetUserDefaultLCID, IsValidLocale, EnumSystemLocalesA, ...):
// GameConfig::user_lang_id, which <ID>_LANG_ID overrides. No host calls
// here, so tests can check any language.

#include <cstdint>
#include <vector>

namespace recomp {

// US English: the language the runtime answers unless the port gives
// another.
inline constexpr uint16_t kLangEnUs = 0x0409;

// A Windows language id (LANGID) of a user's language: a primary language
// in the low 10 bits that is not LANG_NEUTRAL (0), and a sublanguage in
// the top 6 that is not SUBLANG_NEUTRAL (0). A neutral id such as 0x0019
// (LANG_RUSSIAN alone) is never a user's or an interface's language on
// Windows, and games that compare the whole id would not recognise it.
inline constexpr bool IsLangId(uint32_t v) {
  return v <= 0xFFFF && (v & 0x3FF) != 0 && (v >> 10) != 0;
}

// The locale of a language with the default sort order,
// MAKELCID(lang, SORT_DEFAULT): the same number.
inline constexpr uint32_t LcidOf(uint16_t lang) { return lang; }

// "0x0419" or "0X419" (<ID>_LANG_ID): 0x, then 1 to 4 hexadecimal
// digits, a language id per IsLangId. The prefix is required, so that a
// decimal habit ("1049") is an error rather than another language. False
// for anything else, *out unchanged.
bool ParseLangId(const char* text, uint16_t* out);

// The locales the system has: US English and the user's language, both
// with the default sort order, smallest first, each once
// (EnumSystemLocalesA; IsValidLocale accepts these).
std::vector<uint32_t> InstalledLocales(uint16_t user_lang);

}  // namespace recomp
