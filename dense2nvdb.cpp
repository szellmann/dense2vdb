
// std
#include <cassert>
#include <cmath>
#include <cstring>
#include <algorithm>
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
static double g_compressionRate{0.5};
// outdated, we're keeping these for testing though:
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

inline
float2 computeValueRange(const char *input, int3 lower, int3 upper)
{
  float2 valueRange(1e31f,-1e31f);
  for (int z=lower.z; z<upper.z; ++z) {
    for (int y=lower.y; y<upper.y; ++y) {
      for (int x=lower.x; x<upper.x; ++x) {
        const float value = getValue(input, x, y, z);
        valueRange.x = fminf(valueRange.x,value);
        valueRange.y = fmaxf(valueRange.y,value);
      }
    }
  }
  return valueRange;
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

  g_compressionRate = params.compressionRate;
  if (g_compressionRate <= 0.0 || g_compressionRate > 1.0)
    return false;

  g_backgroundValue = params.backgroundValue;
  g_tolerance = params.tolerance;

  return true;
}

// strategy implemented by VDB internally
// keeping this aroudn for testing, at least for now:
static
void compressOpenVDB(const char *input)
{
  if (g_backgroundValue != g_backgroundValue) {
    float2 valueRange = computeValueRange(input, int3(0), g_dims);
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

// our strategy:
static
void compressOpenVDB_v2(const char *input)
{
  double targetCompressionRate = g_compressionRate;

  float2 valueRange = computeValueRange(input, int3(0), g_dims);
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

  // That's the value for maxIndex; the one that occurs most
  // often in our data:
  float maxValue = maxIndex/float(N-1)*(valueRange.y-valueRange.x)+valueRange.x;

  // Distance metrics (how "far away" is a given range from some other value):
  auto distance1 = [](float scalar, float2 scalarRange) // to the closest of the two extrema
  { return fminf(fabsf(scalarRange.x-scalar), fabsf(scalarRange.y-scalar)); };

  auto distance2 = [](float scalar, float2 scalarRange) // to the farthest of the two extrema
  { return fmaxf(fabsf(scalarRange.x-scalar), fabsf(scalarRange.y-scalar)); };

  auto distance3 = [](float scalar, float2 scalarRange) // to the median of the two extrema
  { return fabsf((scalarRange.x+scalarRange.y)*0.5f-scalar); };

  // VDB conversion:
  openvdb::initialize();

  openvdb::math::CoordBBox domain(openvdb::math::Coord(0, 0, 0),
                                  openvdb::math::Coord(g_dims.x, g_dims.y, g_dims.z));

  using FloatTree = openvdb::tree::Tree4<float, 5, 4, 3>::Type;
  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(maxValue);

  // in the following we assume a 5,4,3 layout, i.e., leaf nodes are 2^3 voxel grids
  auto &tree = grid->tree();
  // we want a full domain box, so we set the voxels at
  // the extrema to their actual values and *activate them*:
  tree.addTile(0, openvdb::math::Coord(0, 0, 0), getValue(input, 0, 0, 0), true);
  tree.addTile(0, openvdb::math::Coord(g_dims.x-1, g_dims.y-1, g_dims.z-1),
      getValue(input, g_dims.x-1, g_dims.y-1, g_dims.z-1), true);

  auto div_up = [](int a, int b) { return (a + b - 1) / b; };
  constexpr int brickSizeLog[3] = { 3, 4, 5 }; // from our tree topology
  int level = 1;
  int brickSize = 1<<brickSizeLog[level-1];
  int3 numBricks(div_up(g_dims.x,brickSize),
                 div_up(g_dims.y,brickSize),
                 div_up(g_dims.z,brickSize));

  struct BrickRef { int3 brickID; float2 valueRange; };
  std::vector<BrickRef> brickRefs(numBricks.x*size_t(numBricks.y)*numBricks.z);

  for (int bz=0; bz<numBricks.z; ++bz) {
    for (int by=0; by<numBricks.y; ++by) {
      for (int bx=0; bx<numBricks.x; ++bx) {
        int3 lower = int3(bx,by,bz)*brickSize;
        int3 upper = int3(bx+1,by+1,bz+1)*brickSize;
        upper.x = std::min(upper.x,g_dims.x);
        upper.y = std::min(upper.y,g_dims.y);
        upper.z = std::min(upper.z,g_dims.z);

        size_t brickIndex = bx + by * numBricks.x + bz * size_t(numBricks.x) * numBricks.y;

        brickRefs[brickIndex].brickID = int3(bx,by,bz);
        brickRefs[brickIndex].valueRange = computeValueRange(input, lower, upper);
      }
    }
  }

  std::sort(brickRefs.begin(),
            brickRefs.end(),
            [&](const BrickRef &a, const BrickRef &b)
            { return distance2(maxValue, a.valueRange) < distance2(maxValue, b.valueRange); });

  // std::cout << brickRefs[0].valueRange << '\n';
  // std::cout << brickRefs[1].valueRange << '\n';
  // std::cout << brickRefs.back().valueRange << '\n';

  size_t numBricksToActivate(brickRefs.size() * targetCompressionRate);
  std::cout << "Target compression rate: " << targetCompressionRate << '\n';
  std::cout << "Activating " << numBricksToActivate << " out of " << brickRefs.size()
    << " level-" << (level-1) << " bricks\n";

  for (size_t i=brickRefs.size(); i>=brickRefs.size()-numBricksToActivate; i--) {
    const BrickRef &ref = brickRefs[i];
    int bx = ref.brickID.x;
    int by = ref.brickID.y;
    int bz = ref.brickID.z;
    int3 lower = int3(bx,by,bz)*brickSize;
    int3 upper = int3(bx+1,by+1,bz+1)*brickSize;
    for (int z=lower.z; z<upper.z; ++z) {
      for (int y=lower.y; y<upper.y; ++y) {
        for (int x=lower.x; x<upper.x; ++x) {
          tree.addTile(0, openvdb::math::Coord(x,y,z), getValue(input, x, y, z), true);
        }
      }
    }
  }

  tree.prune();

  std::cout << "activeVoxelCount: " << tree.activeVoxelCount() << '\n';
  //std::cout << "activeTileCount: " << tree.activeTileCount() << '\n';
  std::cout << "Compression achieved: "
    << double(tree.activeVoxelCount()/double(g_dims.x*size_t(g_dims.y)*g_dims.z)) << '\n';

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
      compressOpenVDB_v2(raw_in);
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




