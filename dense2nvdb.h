
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
  int dims[3];
  const char *type;
  double compressionRate;
  // outdated, we're keeping these for testing though:
  float backgroundValue;
  float tolerance;
} d2nvdbParams;

void d2nvdbCompress(const char *raw_in,
                    d2nvdbParams *params,
                    char *nvdb_out,
                    uint64_t *nvdb_size);

#ifdef __cplusplus
}
#endif
