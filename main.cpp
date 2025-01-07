// std
#include <string>
#include <fstream>
#include <vector>
// openvdb
#include <openvdb/openvdb.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/math/Math.h>
// ours
#include "math_.h"

using int3 = math::vec3i;
using float2 = math::vec2f;
using float3 = math::vec3f;

// App state
static openvdb::tools::Dense<float> *g_dense{nullptr};
static std::string g_inFileName;
static std::string g_outFileName;
static int3 g_dims{0,0,0};
static std::string g_type = "float";
static float g_backgroundValue{0.f};
static float g_tolerance{0.f};

static bool parseCommandLine(int argc, char **argv)
{
  for (int i = 1; i < argc; i++) {
    const std::string arg = argv[i];
    if (arg == "-o")
      g_outFileName = argv[++i];
    else if (arg == "-dims") {
      g_dims.x = atoi(argv[++i]);
      g_dims.y = atoi(argv[++i]);
      g_dims.z = atoi(argv[++i]);
    } else if (arg == "-type")
      g_type = argv[++i];
    else if (arg == "-tolerance")
      g_tolerance = atof(argv[++i]);
    else if (arg[0] != '-')
      g_inFileName = arg;
    else return false;
  }

  return true;
}

static bool validateInput()
{
  if (g_dims.x <= 0 || g_dims.y <= 0 || g_dims.z <= 0 ||
    g_inFileName.empty() || g_outFileName.empty())
    return false;

  return true;
}

static void printUsage()
{
  std::cerr << "./app in.raw -dims w h d -type {byte|short|float} [-tolerance <float-val>] -o out.vdb\n";
}

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

int main(int argc, char **argv)
{
  if (!parseCommandLine(argc, argv)) {
    printUsage();
    exit(1);
  }

  if (!validateInput()) {
    printUsage();
    exit(1);
  }

  std::ifstream in(g_inFileName);

  if (!in.good()) {
    printUsage();
    exit(1);
  }

  std::vector<char> input;

  if (g_type == "byte") {
    input.resize(sizeof(char) * g_dims.x * size_t(g_dims.y) * g_dims.z);
  }
  else if (g_type == "float") {
    input.resize(sizeof(float) * g_dims.x * size_t(g_dims.y) * g_dims.z);
  }

  if (input.empty()) {
    printUsage();
    exit(1);
  }

  in.read(input.data(), input.size());

  std::cout << "Loading file: " << g_inFileName << '\n';
  std::cout << "Dims: " << g_dims.x << ':' << g_dims.y << ':' << g_dims.z << '\n';
  std::cout << "Type: " << g_type << '\n';

  openvdb::initialize();

  openvdb::math::CoordBBox domain(openvdb::math::Coord(0, 0, 0),
                                  openvdb::math::Coord(g_dims.x, g_dims.y, g_dims.z));

  g_dense = new openvdb::tools::Dense<float>(domain, 0.f);

  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        openvdb::math:: Coord ijk(x,y,z);
        float value = 0.f;
        if (g_type == "byte") {
          unsigned char *inVoxels = (unsigned char *)input.data();
          value = inVoxels[x+y*g_dims.x+z*size_t(g_dims.x)*g_dims.y]/255.f;
        }
        else if (g_type == "float") {
          float *inVoxels = (float *)input.data();
          value = inVoxels[x+y*g_dims.x+z*size_t(g_dims.x)*g_dims.y];
        }
        g_dense->setValue(ijk, value);
      }
    }
  }

  FloatRule rule(g_backgroundValue, g_tolerance);
  openvdb::FloatTree::Ptr result
      = openvdb::tools::extractSparseTree(*g_dense, rule, g_backgroundValue);
  result->prune();

  std::cout << "Extracted sparse tree:\n";
  std::cout << "activeVoxelCount: " << result->activeVoxelCount() << '\n';
  std::cout << "leafCount: " << result->leafCount() << '\n';

  openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create(result);

  openvdb::CoordBBox bbox;
  auto iter = grid->cbeginValueOn();
  iter.getBoundingBox(bbox);
  std::cout << "Bounding box: " << bbox << std::endl;

  std::cout << "Writing to file: " << g_outFileName << '\n';

  // store:
  openvdb::io::File file(g_outFileName);
  // Add the grid pointer to a container.
  openvdb::GridPtrVec grids;
  grids.push_back(grid);
  // Write out the contents of the container.
  file.write(grids);
  file.close();
  return 0;
}



