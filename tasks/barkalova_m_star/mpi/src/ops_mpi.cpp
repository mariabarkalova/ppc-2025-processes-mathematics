#include "barkalova_m_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <cstddef>
#include <utility>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"

namespace barkalova_m_star {

BarkalovaMStarMPI::BarkalovaMStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BarkalovaMStarMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  const auto &input = GetInput();
  if (input.source < 0 || input.source >= size) {
    return false;
  }
  if (input.dest < 0 || input.dest >= size) {
    return false;
  }
  return true;
}

bool BarkalovaMStarMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

namespace {

void ToMyself(int rank, int source, const std::vector<int> &source_data, std::vector<int> &output) {
  if (rank == source) {
    output = source_data;
  }
}

void ProcSource(int source, size_t data_size, const std::vector<int> &source_data) {
  if (source != 0) {
    MPI_Send(source_data.data(), static_cast<int>(data_size), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

void ProcDest(int dest, size_t data_size, std::vector<int> &output) {
  std::vector<int> buff(data_size);
  if (dest != 0) {
    MPI_Status status;
    MPI_Recv(buff.data(), static_cast<int>(data_size), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
  }
  output = std::move(buff);
}

void Center(int source, int dest, size_t data_size, const std::vector<int> &source_data, std::vector<int> &output) {
  if (source != 0) {
    std::vector<int> buff(data_size);
    MPI_Status status;
    MPI_Recv(buff.data(), static_cast<int>(data_size), MPI_INT, source, 0, MPI_COMM_WORLD, &status);
    if (dest != 0) {
      MPI_Send(buff.data(), static_cast<int>(data_size), MPI_INT, dest, 0, MPI_COMM_WORLD);
    } else {
      output = std::move(buff);
    }
  } else {
    MPI_Send(source_data.data(), static_cast<int>(data_size), MPI_INT, dest, 0, MPI_COMM_WORLD);
  }
}

void ToAnother(int rank, int source, int dest, size_t data_size, const std::vector<int> &source_data,
               std::vector<int> &output) {
  if (rank == dest || (rank == 0 && dest == 0)) {
    output.resize(data_size);
  }

  if (rank == 0) {
    Center(source, dest, data_size, source_data, output);
  } else if (rank == source) {
    ProcSource(source, data_size, source_data);
  } else if (rank == dest) {
    ProcDest(dest, data_size, output);
  }
}
}  // namespace

bool BarkalovaMStarMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();
  int source = input.source;
  int dest = input.dest;

  if (size == 1) {
    GetOutput() = GetInput().data;
    return true;
  }
  const std::vector<int> &data = input.data;
  size_t data_size = data.size();

  if (rank == dest || (rank == 0 && dest == 0)) {
    GetOutput().resize(data_size);
  } else {
    GetOutput() = {};
  }

  if (source == dest) {
    ToMyself(rank, source, input.data, GetOutput());
  } else {
    ToAnother(rank, source, dest, data_size, input.data, GetOutput());
  }

  return true;
}

bool BarkalovaMStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_star
