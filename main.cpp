// std
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
#include "math_.h"
#include "nanovdb_convert.h"
#include "dense2nvdb.h"

using int3 = math::vec3i;
using float2 = math::vec2f;
using float3 = math::vec3f;

// App state
static openvdb::tools::Dense<float> *g_dense{nullptr};
static std::string g_inFileName;
static std::string g_outFileName;
static int3 g_dims{0,0,0};
static std::string g_type = "float";
static double g_compressionRate{0.5};
// outdated, we're keeping these for testing though:
static float g_backgroundValue{NAN};
static float g_tolerance{NAN};

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
    else if (arg == "-c" || arg == "compressionRate")
      g_compressionRate = atof(argv[++i]);
    /* deprecated:*/
    else if (arg == "-background" || arg == "-bg")
      g_backgroundValue = atof(argv[++i]);
    else if (arg == "-tolerance")
      g_tolerance = atof(argv[++i]);
    /* END deprecated */
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
  std::cerr << "./app in.raw -dims w h d -type {byte|short|float} [-background <float-val> -tolerance <float-val>] -o out.vdb\n";
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

int main(int argc, char **argv)
{
  LOG_START;

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

  LOG_OMP(input_resize);

  if (input.empty()) {
    printUsage();
    exit(1);
  }

  in.read(input.data(), input.size());
  LOG_OMP(in_read);

  std::cout << "Loading file: " << g_inFileName << '\n';
  std::cout << "Dims: " << g_dims.x << ':' << g_dims.y << ':' << g_dims.z << '\n';
  std::cout << "Type: " << g_type << '\n';

  d2nvdbParams parms;
  parms.dims[0] = g_dims.x;
  parms.dims[1] = g_dims.y;
  parms.dims[2] = g_dims.z;
  parms.type = g_type.c_str();
  parms.compressionRate = g_compressionRate;

  uint64_t bufferSize;
#ifdef EXPORT_VDB
  d2nvdbCompress(input.data(), &parms, nullptr, &bufferSize, g_outFileName.c_str());
#else  
  d2nvdbCompress(input.data(), &parms, nullptr, &bufferSize);
#endif

  LOG_OMP(d2nvdbCompress1);

  constexpr size_t alignment = 32;
  char* alignedBuffer = (char*)std::aligned_alloc(alignment, bufferSize);

  d2nvdbCompress(input.data(), &parms, alignedBuffer, &bufferSize);

  LOG_OMP(d2nvdbCompress2);

  auto nvdbBuffer = nanovdb::HostBuffer::createFull(bufferSize, alignedBuffer);
  nanovdb::GridHandle<nanovdb::HostBuffer> handle = std::move(nvdbBuffer);

  LOG_OMP(GridHandle);

  std::ofstream os(g_outFileName, std::ios::out | std::ios::binary);
  nanovdb::io::Codec       codec = nanovdb::io::Codec::NONE;// compression codec for the file
  nanovdb::io::writeGrid(os, handle, codec);

  std::free(alignedBuffer);

  LOG_OMP(writeGrid);

  return 0;
}



