
#pragma once

#include <stdint.h>

#if defined(USE_LOG) || defined(USE_OPENMP)
# include <omp.h>
#endif

#ifdef USE_LOG
# define LOG_START double _t = omp_get_wtime();
# define LOG_OMP(name) printf("LOG: %s: %f [s](%s, %d)\n", #name, omp_get_wtime() - _t, __FUNCTION__, __LINE__);
#else
# define LOG_START
# define LOG_OMP(name)
#endif

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
                    uint64_t *nvdb_size,
                    const char* filename = nullptr);

#ifdef __cplusplus
}
#endif
