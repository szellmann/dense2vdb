// std
#include <cfloat>
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
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

#ifdef USE_MPI
# include <mpi.h>
#endif

#ifdef _MSC_VER
# include <malloc.h>  // MSVC uses _aligned_malloc
#endif

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

static void* aligned_allocate(size_t alignment, size_t size) {
#ifdef _MSC_VER
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

#ifdef USE_MPI
const size_t MAX_MPI_TRANSFER = std::numeric_limits<int>::max(); // 2GB limit per MPI send/recv
static void read_binary_file(const std::string& filename, std::vector<char>& data, size_t offset, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    file.seekg(offset, std::ios::beg);    
    file.read(data.data(), size);
}
#endif

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
  float v_norm = (value-minValue)/(maxValue-minValue);
  if (v_norm < 0.f)
    v_norm = 0.f;
  if (v_norm > 1.f)
    v_norm = 1.f;
    
  return v_norm;
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

  float minValue = FLT_MAX;
  float maxValue = -FLT_MAX;
  float minVDB = FLT_MAX;
  float maxVDB = -FLT_MAX; 

  // compute min/max's
#ifdef USE_OPENMP
# pragma omp parallel for reduction(min: minValue, minVDB) reduction(max: maxValue, maxVDB)
#endif  
  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        float value0 = getValue(input,x,y,z);
        minValue = fminf(minValue,value0);
        maxValue = fmaxf(maxValue,value0);

        float value1 = getValue(comp,x,y,z);
        minVDB = fminf(minVDB,value1);
        maxVDB = fmaxf(maxVDB,value1);
      }
    }
  }

  res.minValue = minValue;
  res.maxValue = maxValue;
  res.minVDB = minVDB;
  res.maxVDB = maxVDB;  

  double sumSquared{0.0};
  double sumSquaredErr{0.0};

#ifdef USE_OPENMP
# pragma omp parallel for reduction(+: sumSquared, sumSquaredErr)
#endif  
  for (int z=0; z<g_dims.z; ++z) {
    for (int y=0; y<g_dims.y; ++y) {
      for (int x=0; x<g_dims.x; ++x) {
        float value0 = mapValue(getValue(input,x,y,z), res.minValue, res.maxValue);
        float value1 = mapValue(getValue(comp,x,y,z), res.minValue, res.maxValue);
        double sqr = double(value0) * double(value0);
        sumSquared += sqr;
        double diff = double(value0) - double(value1);
        sumSquaredErr += diff * diff;
      }
    }
  }
  size_t N = g_dims.x * size_t(g_dims.y) * g_dims.z;
  res.mse = sumSquaredErr / N;
  double signalMean = sumSquared / N;
  double noiseMean = res.mse;
  if (noiseMean == 0.0) res.snr = INFINITY;
  //else res.snr = 20*log10(sqrt(signalMean)/sqrt(noiseMean));
  else res.snr = 20*log10(1.0f/sqrt(res.mse));
  
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
  LOG_START;

#ifdef USE_MPI
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif  

  if (!parseCommandLine(argc, argv)) {
    printUsage();
#ifdef USE_MPI
    MPI_Finalize();
#endif
    exit(1);
  }

  if (!validateInput()) {
    printUsage();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
    exit(1);
  }

  if (g_verbose) {
    std::cout << "OpenVDB Version: "  << openvdb::OPENVDB_LIBRARY_VERSION << '\n';
  }

  std::vector<char> input;
  size_t file_size = 0;

  if (g_type == "byte") {
    file_size = sizeof(char) * g_dims.x * size_t(g_dims.y) * g_dims.z;
  }
  else if (g_type == "float") {
    file_size = sizeof(float) * g_dims.x * size_t(g_dims.y) * g_dims.z;
  }

  if (file_size == 0) {
    printUsage();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
    exit(1);
  }

#ifdef USE_MPI
  // Compute local chunk size for each process
  size_t chunk_size = file_size / size;
  size_t remainder = file_size % size;
  size_t local_size = (rank < remainder) ? (chunk_size + 1) : chunk_size;
  size_t offset = rank * chunk_size + std::min<size_t>(rank, remainder);

  // Rank 0 receives all chunks asynchronously
  if (rank == 0) {
      input.resize(file_size);
      LOG_OMP(input_resize);
      read_binary_file(g_inFileName, input, offset, local_size);
      LOG_OMP(read_binary_file);

      std::vector<MPI_Request> requests(size - 1);

      // Receive data in chunks using multiple MPI_Irecv calls if needed
      size_t recv_offset = local_size;
      for (int src = 1; src < size; ++src) {
          size_t recv_size = (src < remainder) ? (chunk_size + 1) : chunk_size;
          size_t num_chunks = (recv_size + MAX_MPI_TRANSFER - 1) / MAX_MPI_TRANSFER; // Number of 2GB chunks

          for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
              size_t chunk_offset = recv_offset + chunk * MAX_MPI_TRANSFER;
              size_t chunk_size = std::min(MAX_MPI_TRANSFER, recv_size - chunk * MAX_MPI_TRANSFER);

              MPI_Irecv(input.data() + chunk_offset, static_cast<int>(chunk_size),
                  MPI_CHAR, src, chunk, MPI_COMM_WORLD, &requests[src - 1]);
          }
          recv_offset += recv_size;
      }

      // Wait for all receives to complete
      MPI_Waitall(size - 1, requests.data(), MPI_STATUSES_IGNORE);
      LOG_OMP(MPI_Waitall);
      
      std::cout << "File size: " << input.size() << " bytes" << std::endl;
  }
  else {
      // Each process reads its chunk
      std::vector<char> local_data;
      local_data.resize(local_size);
      read_binary_file(g_inFileName, local_data, offset, local_size);

      // Send data in chunks using multiple MPI_Isend calls if needed
      MPI_Request request;
      size_t num_chunks = (local_size + MAX_MPI_TRANSFER - 1) / MAX_MPI_TRANSFER; // Number of 2GB chunks

      for (size_t chunk = 0; chunk < num_chunks; ++chunk) {
          size_t chunk_offset = chunk * MAX_MPI_TRANSFER;
          size_t chunk_size = std::min(MAX_MPI_TRANSFER, local_size - chunk_offset);

          MPI_Isend(local_data.data() + chunk_offset, static_cast<int>(chunk_size),
              MPI_CHAR, 0, chunk, MPI_COMM_WORLD, &request);
      }

      MPI_Wait(&request, MPI_STATUS_IGNORE);

      MPI_Barrier(MPI_COMM_WORLD);
      MPI_Finalize();

      return 0;
  }
    
#else
  std::ifstream in(g_inFileName);

  if (!in.good()) {
    printUsage();
#ifdef USE_MPI
    MPI_Finalize();
#endif    
    exit(1);
  }

  input.resize(file_size);
  LOG_OMP(input_resize);  
  in.read(input.data(), input.size());
  LOG_OMP(in_read);
#endif    

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

#if defined(EXPORT_VDB) || defined(EXPORT_HISTOGRAM)
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

  LOG_OMP(writeGrid);

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif

  return 0;
}



