// std
#include <cmath>
#include <cstring>
#include <string>
#include <iostream>
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

#ifdef USE_MPI
# include <mpi.h>
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
// outdated, we're keeping these for testing though:
static float g_backgroundValue{NAN};
static float g_tolerance{NAN};

#ifdef USE_MPI
void read_binary_file(const std::string& filename, std::vector<char>& data, size_t offset, size_t size) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file: " << filename << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    file.seekg(offset, std::ios::beg);
    data.resize(size);
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
    // Compute local chunk size
    size_t chunk_size = file_size / size;
    size_t remainder = file_size % size;
    size_t local_size = (rank < remainder) ? (chunk_size + 1) : chunk_size;
    size_t offset = rank * chunk_size + std::min<size_t>(rank, remainder);

    // Each process reads its chunk
    std::vector<char> local_data;
    read_binary_file(g_inFileName, local_data, offset, local_size);

    LOG_OMP(in_read);

    // Gather sizes at rank 0
    std::vector<int> recv_counts(size);
    std::vector<int> displacements(size);
    int local_size_int = static_cast<int>(local_size);
    
    MPI_Gather(&local_size_int, 1, MPI_INT, recv_counts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displacements[0] = 0;
        for (int i = 1; i < size; ++i) {
            displacements[i] = displacements[i - 1] + recv_counts[i - 1];
        }
    }

    LOG_OMP(displacements);

    // Rank 0 allocates a buffer for the complete file
    if (rank == 0) {
        input.resize(file_size);
    }

    LOG_OMP(input_resize);

    // Gather all data at rank 0
    MPI_Gatherv(local_data.data(), local_size_int, MPI_CHAR,
                input.data(), recv_counts.data(), displacements.data(),
                MPI_CHAR, 0, MPI_COMM_WORLD);

    LOG_OMP(MPI_Gatherv);
    
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

#ifdef USE_MPI
    MPI_Finalize();
#endif

  return 0;
}



