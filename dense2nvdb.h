// ======================================================================== //
// Copyright 2022-2025 Stefan Zellmann                                      //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

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
