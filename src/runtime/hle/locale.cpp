#include "runtime/hle/locale.h"

namespace recomp {

bool ParseLangId(const char* text, uint16_t* out) {
  if (!text) return false;
  if (text[0] != '0' || (text[1] != 'x' && text[1] != 'X')) return false;
  const char* p = text + 2;
  uint32_t v = 0;
  int digits = 0;
  for (; *p; p++, digits++) {
    int d;
    if (*p >= '0' && *p <= '9')
      d = *p - '0';
    else if (*p >= 'a' && *p <= 'f')
      d = *p - 'a' + 10;
    else if (*p >= 'A' && *p <= 'F')
      d = *p - 'A' + 10;
    else
      return false;
    if (digits == 4) return false;
    v = v * 16 + uint32_t(d);
  }
  if (!digits || !IsLangId(v)) return false;
  *out = uint16_t(v);
  return true;
}

std::vector<uint32_t> InstalledLocales(uint16_t user_lang) {
  const uint32_t en = LcidOf(kLangEnUs), user = LcidOf(user_lang);
  if (user == en) return {en};
  if (user < en) return {user, en};
  return {en, user};
}

}  // namespace recomp
