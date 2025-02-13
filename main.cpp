// std
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <fstream>
#include <vector>
// openvdb
#include <openvdb/openvdb.h>
#include <openvdb/tools/DenseSparseTools.h>
#include <openvdb/math/Math.h>
// zfp
#ifdef WITH_ZFP
#include <zfp.h>
#endif
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
static bool g_stats{false};
static bool g_zfp{false};
static bool g_verbose{false};
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
    else if (arg == "-stats")
      g_stats = true;
    else if (arg == "-zfp")
      g_zfp = true;
    /* deprecated:*/
    else if (arg == "-background" || arg == "-bg")
      g_backgroundValue = atof(argv[++i]);
    else if (arg == "-tolerance")
      g_tolerance = atof(argv[++i]);
    else if (arg == "-v" || arg == "-verbose")
      g_verbose = true;
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
  std::cerr << "./app in.raw -dims w h d -type {byte|short|float} [-c <float-val> -stats] -o out.vdb\n";
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

inline
float getValue(const float *data, int x, int y, int z)
{
  return data[x+y*g_dims.x+z*size_t(g_dims.x)*g_dims.y];
}

inline
float getValue(nanovdb::NanoGrid<float> *nvdb, int x, int y, int z)
{
  auto acc = nvdb->getAccessor();
  return acc.getValue(nanovdb::Coord(x,y,z));
}

inline
float mapValue(float value, float minValue, float maxValue)
{
  return (value-minValue)/(maxValue-minValue);
}

struct Stats
{
  float minValue{FLT_MAX}, maxValue{-FLT_MAX};
  float minVDB{FLT_MAX}, maxVDB{-FLT_MAX};
  double mse{0.0};
  double snr{0.0};
};

template <typename Compressed>
Stats computeStats(const char *input, Compressed comp)
{
  Stats res;

  // compute min/max's
  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        float value0 = getValue(input,x,y,z);
        res.minValue = fminf(res.minValue,value0);
        res.maxValue = fmaxf(res.maxValue,value0);

        float value1 = getValue(comp,x,y,z);
        res.minVDB = fminf(res.minVDB,value1);
        res.maxVDB = fmaxf(res.maxVDB,value1);
      }
    }
  }

  double sumSquared{0.0};
  double sumSquaredErr{0.0};
  double signalMean{0.0};
  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        float value0 = mapValue(getValue(input,x,y,z), res.minValue, res.maxValue);
        float value1 = mapValue(getValue(comp,x,y,z), res.minVDB, res.maxVDB);
        signalMean += value0;
        double sqr = double(value0) * double(value0);
        sumSquared += sqr;
        double diff = double(value0) - double(value1);
        sumSquaredErr += diff * diff;
      }
    }
  }
  size_t N = g_dims.x * size_t(g_dims.y) * g_dims.z;
  signalMean /= N;
  res.mse = sumSquaredErr / N;
  double signalMean2 = sumSquared / N;
  double noiseMean = res.mse;
  if (noiseMean == 0.0) res.snr = INFINITY;
  //else res.snr = 20*log10(sqrt(signalMean2)/sqrt(noiseMean));
  else res.snr = signalMean / sqrt(noiseMean);
  return res;
}

#ifdef WITH_ZFP
std::vector<uchar> zfpCompress(std::vector<float> &input)
{
  uint dims = 3;
  zfp_type type = zfp_type_float;
  zfp_field* field = zfp_field_3d(input.data(), type, g_dims.x, g_dims.y, g_dims.z); 
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, g_compressionRate*8/*8 bits*/, type, dims, 0);
  //zfp_stream_set_precision(zfp, precision, type);
  //zfp_stream_set_accuracy(zfp, tolerance, type);
  size_t bufsize = zfp_stream_maximum_size(zfp, field);
  std::vector<uchar> buffer(bufsize);
  bitstream* stream = stream_open(buffer.data(), bufsize);
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  std::cout << "compress...\n";
  size_t size = zfp_compress(zfp, field);
  buffer.resize(size); // adjust to actual size
  return buffer;
}

std::vector<float> zfpDecompress(std::vector<uchar> &buffer)
{
  uint dims = 3;
  std::vector<float> result(g_dims.x*size_t(g_dims.y)*g_dims.z);
  zfp_type type = zfp_type_float;
  zfp_field* field = zfp_field_3d(result.data(), type, g_dims.x, g_dims.y, g_dims.z); 
  zfp_stream* zfp = zfp_stream_open(NULL);
  zfp_stream_set_rate(zfp, g_compressionRate*8/* bits*/, type, dims, 0);
  bitstream* stream = stream_open(buffer.data(), buffer.size());
  zfp_stream_set_bit_stream(zfp, stream);
  zfp_stream_rewind(zfp);
  size_t size = zfp_decompress(zfp, field);
  return result;
}
#endif

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

  if (g_verbose) {
    std::cout << "OpenVDB Version: "  << openvdb::OPENVDB_LIBRARY_VERSION << '\n';
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

  if (g_zfp) { // ZFP compression (only for comparison, output file is *not* compressed!):
#ifdef WITH_ZFP
    // convert to float:
    std::vector<float> inputf;
    for (int z=0; z<g_dims.z; ++z) {
      for (int y=0; y<g_dims.y; ++y) {
        for (int x=0; x<g_dims.x; ++x) {
          inputf.push_back(getValue(input.data(),x,y,z));
        }
      }
    }
    auto buffer = zfpCompress(inputf);
    std::cout << "Compression achieved: "
      << buffer.size()/double(g_dims.x*size_t(g_dims.y)*g_dims.z) << '\n';
    auto decomp = zfpDecompress(buffer);
    if (1 || g_stats) { // stats, that's what this test is for...
      auto s = computeStats(input.data(), decomp.data());
      std::cout << "min/max (in) ....: [" << s.minValue << ',' << s.maxValue << "]\n";
      std::cout << "min/max (out) ...: [" << s.minVDB << ',' << s.maxVDB << "]\n";
      std::cout << "MSE .............: " << s.mse << '\n';
      std::cout << "SNR .............: " << s.snr << '\n';
    }

    std::ofstream os(g_outFileName, std::ios::out | std::ios::binary);
    os.write((const char *)decomp.data(), g_dims.x*size_t(g_dims.y)*g_dims.z * sizeof(float));
#endif
  } else { // VDB compression ("ours"):
    d2nvdbParams parms;
    parms.dims[0] = g_dims.x;
    parms.dims[1] = g_dims.y;
    parms.dims[2] = g_dims.z;
    parms.type = g_type.c_str();
    parms.compressionRate = g_compressionRate;

    uint64_t bufferSize;
    d2nvdbCompress(input.data(), &parms, nullptr, &bufferSize);

    constexpr size_t alignment = 32;
    char* alignedBuffer = (char*)std::aligned_alloc(alignment, bufferSize);

    d2nvdbCompress(input.data(), &parms, alignedBuffer, &bufferSize);

    auto nvdbBuffer = nanovdb::HostBuffer::createFull(bufferSize, alignedBuffer);
    nanovdb::GridHandle<nanovdb::HostBuffer> handle = std::move(nvdbBuffer);

    if (g_stats) {
      auto s = computeStats(input.data(), handle.grid<float>());
      std::cout << "min/max (in) ....: [" << s.minValue << ',' << s.maxValue << "]\n";
      std::cout << "min/max (out) ...: [" << s.minVDB << ',' << s.maxVDB << "]\n";
      std::cout << "MSE .............: " << s.mse << '\n';
      std::cout << "SNR .............: " << s.snr << '\n';
    }

    std::ofstream os(g_outFileName, std::ios::out | std::ios::binary);
    nanovdb::io::Codec       codec = nanovdb::io::Codec::NONE;// compression codec for the file
    nanovdb::io::writeGrid(os, handle, codec);

    std::free(alignedBuffer);
  }

  return 0;
}



