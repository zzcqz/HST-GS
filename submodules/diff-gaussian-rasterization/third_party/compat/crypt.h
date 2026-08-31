// Minimal compatibility header for environments without <crypt.h>.
// This project does not call crypt() directly; the header is only needed
// because some Python headers include <crypt.h> unconditionally.
#pragma once

#ifdef __cplusplus
extern "C" {
#endif

char* crypt(const char* key, const char* salt);

#ifdef __cplusplus
}
#endif

