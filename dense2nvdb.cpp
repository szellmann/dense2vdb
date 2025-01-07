
// std
#include <cassert>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
// openvdb
#include <openvdb/openvdb.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/math/Math.h>
// ours
#include "dense2nvdb.h"
#include "math_.h"
#include "nanovdb_convert.h"

using int3 = math::vec3i;
using float2 = math::vec2f;
using float3 = math::vec3f;

// App state
static openvdb::tools::Dense<float> *g_dense{nullptr};
static openvdb::GridPtrVec g_sparseGrids;
static std::vector<uint8_t> g_nvdbGridData;
static std::string g_inFileName;
static std::string g_outFileName;
static int3 g_dims{0,0,0};
static std::string g_type = "";
static float g_backgroundValue{NAN};
static float g_tolerance{NAN};

//-----------------------------------------------------------------------------
// Impl:
//-----------------------------------------------------------------------------

class FloatRule
{
public:
  typedef openvdb::FloatTree            ResultTreeType;
  typedef ResultTreeType::LeafNodeType  ResultLeafNodeType;

  typedef float                                  ResultValueType;
  typedef float                                  DenseValueType;

  FloatRule(const DenseValueType &value, const DenseValueType &tolerance = DenseValueType(0.0))
    : mMaskValue(value),
      mTolerance(tolerance)
  {}

  template <typename IndexOrCoord>
  void operator()(const DenseValueType& a, const IndexOrCoord& offset,
                  ResultLeafNodeType* leaf) const
  {
    if (a <= mMaskValue-mTolerance || a >= mMaskValue+mTolerance) {
      leaf->setValueOn(offset, a);
    }
  }

private:
    const DenseValueType mMaskValue;
    const DenseValueType mTolerance;
};

inline
float getValue(const char *input, int x, int y, int z)
{
  float value = 0.f;
  if (g_type == "byte") {
    unsigned char *inVoxels = (unsigned char *)input;
    value = inVoxels[x+y*g_dims.x+z*size_t(g_dims.x)*g_dims.y]/255.f;
  }
  else if (g_type == "float") {
    float *inVoxels = (float *)input;
    value = inVoxels[x+y*g_dims.x+z*size_t(g_dims.x)*g_dims.y];
  }
  return value;
}

static
bool populateState(const d2nvdbParams &params)
{
  g_dims.x = params.dims[0];
  g_dims.y = params.dims[1];
  g_dims.z = params.dims[2];

  if (g_dims.x <= 0 || g_dims.y <= 0 || g_dims.z <= 0)
    return false;

  if (std::string(params.type) == "byte" ||
      std::string(params.type) == "short" ||
      std::string(params.type) == "float") {
    g_type = std::string(params.type);
  }

  if (g_type.empty())
    return false;

  g_backgroundValue = params.backgroundValue;
  g_tolerance = params.tolerance;

  return true;
}

static
void compressOpenVDB(const char *input)
{
  if (g_backgroundValue != g_backgroundValue) {
    // compute value range:
    float2 valueRange(1e31f,-1e31f);
    for (int z=0; z<g_dims.z; ++z) {
      for (int y=0; y<g_dims.y; ++y) {
        for (int x=0; x<g_dims.x; ++x) {
          const float value = getValue(input, x, y, z);
          valueRange.x = fminf(valueRange.x,value);
          valueRange.y = fmaxf(valueRange.y,value);
        }
      }
    }
    std::cout << "Computed value range: " << valueRange << '\n';

    // compute histogram:
    #define N (1<<10)
    uint64_t counts[N];
    std::memset(counts,0,sizeof(counts));

    for (int z=0; z<g_dims.z; ++z) {
      for (int y=0; y<g_dims.y; ++y) {
        for (int x=0; x<g_dims.x; ++x) {
          float value = getValue(input, x, y, z);
          value -= valueRange.x;
          value /= valueRange.y-valueRange.x;
          value *= N-1;
          counts[int(value)]++;
        }
      }
    }

    int maxIndex = -1;
    uint64_t maxCount = 0;
    for (int i=0; i<N; ++i) {
      if (counts[i] > maxCount) {
        maxIndex = i;
        maxCount = counts[i];
      }
    }

    // tolerance depending on neighborhood:
    // (this doesn't work so well yet; probably b/c our
    // histograms are spiky, we should perhaps project the values
    // using splatting or so?! Altenratively, use -tolerance on the cmdline!)
    if (g_tolerance != g_tolerance) {
      int stepsL=0, stepsR=0;
      const float sensitivity = 0.9f;
      for (int i=maxIndex; i>=0; --i) {
        if (float(counts[i]) < sensitivity*maxCount) break;
        stepsL++;
      }

      for (int i=maxIndex; i<N; ++i) {
        if (float(counts[i]) < sensitivity*maxCount) break;
        stepsR++;
      }
   
      // reset maxIndex to median between L/R:
      maxIndex = ((maxIndex-stepsL)+(maxIndex+stepsR))/2;
      float valueMedian = maxIndex/float(N-1)*(valueRange.y-valueRange.x)+valueRange.x;
      float valueL = (maxIndex-stepsL)/float(N-1)*(valueRange.y-valueRange.x)+valueRange.x;
      g_tolerance = valueMedian-valueL;
    }

    // value occurring most frequenctly in the data set:
    float maxValue = maxIndex/float(N-1)*(valueRange.y-valueRange.x)+valueRange.x;
    g_backgroundValue = maxValue;
  }

  if (g_tolerance != g_tolerance) {
    g_tolerance = 0.f;
  }

  // VDB conversion:
  openvdb::initialize();

  openvdb::math::CoordBBox domain(openvdb::math::Coord(0, 0, 0),
                                  openvdb::math::Coord(g_dims.x, g_dims.y, g_dims.z));

  g_dense = new openvdb::tools::Dense<float>(domain, 0.f);

  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        openvdb::math:: Coord ijk(x,y,z);
        const float value = getValue(input, x, y, z);
        g_dense->setValue(ijk, value);
      }
    }
  }

  FloatRule rule(g_backgroundValue, g_tolerance);
  openvdb::FloatTree::Ptr result
      = openvdb::tools::extractSparseTree(*g_dense, rule, g_backgroundValue);
  result->prune();

  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(result);
  g_sparseGrids.push_back(grid);
}

//-----------------------------------------------------------------------------
// API:
//-----------------------------------------------------------------------------

void d2nvdbCompress(const char *raw_in,
                    d2nvdbParams *params,
                    char *nvdb_out,
                    uint64_t *nvdb_size)
{
  if (!nvdb_out) {
    // that's the best we can do for now, is to compress
    // the data, store a copy of it, tell the user the size
    // so they can create a scratch buffer. The later call
    // to compress will populate that buffer _from the compressed
    // NVDB_!
    if (populateState(*params)) {
      compressOpenVDB(raw_in);
      if (!g_sparseGrids.empty()) {
        auto handle = nanovdb_convert(&g_sparseGrids);
        g_nvdbGridData.resize(handle.buffer().size());
        memcpy(g_nvdbGridData.data(),
               handle.buffer().data(),
               handle.buffer().size());
        assert(nvdb_size);
        *nvdb_size = (uint64_t)g_nvdbGridData.size();
      }
    }
  }
  else {
    memcpy(nvdb_out,g_nvdbGridData.data(),g_nvdbGridData.size());
  }
}




