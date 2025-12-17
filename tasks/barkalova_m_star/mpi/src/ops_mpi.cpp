/*
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_star {

BarkalovaMStarMPI::BarkalovaMStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BarkalovaMStarMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool BarkalovaMStarMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool BarkalovaMStarMPI::RunImpl() {
  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() /= num_threads;
  } else {
    int counter = 0;
    for (int i = 0; i < num_threads; i++) {
      counter++;
    }

    if (counter != 0) {
      GetOutput() /= counter;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}

bool BarkalovaMStarMPI::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace barkalova_m_star
*/

#include "barkalova_m_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <chrono>  //посмореть
#include <climits>
#include <cstdint>
#include <thread>
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

void HandleSameSourceDestination(int rank, int source, size_t data_size, const std::vector<int> &source_data,
                                 std::vector<int> &output) {
  if (rank == source) {
    output = source_data;
  }
  output.resize(data_size);
  MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, source, MPI_COMM_WORLD);
}

void ProcSource(int source, size_t data_size, const std::vector<int> &source_data) {
  if (source != 0) {
    MPI_Send(source_data.data(), static_cast<int>(data_size), MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
}

void ProcDest(int dest, size_t data_size, std::vector<int> &output) {
  if (dest != 0) {
    std::vector<int> buff(data_size);
    MPI_Status status;
    MPI_Recv(buff.data(), static_cast<int>(data_size), MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
    output = std::move(buff);
  }
  // output = buff;
  //  Если dest == 0, данные уже должны быть в output после ProcessZeroRouting
}

void ProcessZeroRouting(int source, int dest, size_t data_size, const std::vector<int> &source_data,
                        std::vector<int> &output) {
  if (source != 0) {
    std::vector<int> buff(data_size);
    MPI_Status status;
    MPI_Recv(buff.data(), static_cast<int>(data_size), MPI_INT, source, 0, MPI_COMM_WORLD, &status);
    if (dest != 0) {
      MPI_Send(buff.data(), static_cast<int>(data_size), MPI_INT, dest, 0, MPI_COMM_WORLD);
    } else {
      // output = buff;
      output = std::move(buff);
    }
  } else {
    MPI_Send(source_data.data(), static_cast<int>(data_size), MPI_INT, dest, 0, MPI_COMM_WORLD);
  }
}

void HandleDifferentSourceDestination(int rank, int source, int dest, size_t data_size,
                                      const std::vector<int> &source_data, std::vector<int> &output) {
  output.resize(data_size);
  if (rank == 0) {
    ProcessZeroRouting(source, dest, data_size, source_data, output);
  } else if (rank == source) {
    ProcSource(source, data_size, source_data);
  } else if (rank == dest) {
    ProcDest(dest, data_size, output);
  }
  // для чего?
  MPI_Bcast(output.data(), static_cast<int>(data_size), MPI_INT, dest, MPI_COMM_WORLD);
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

  // подумать
  if (size < 3) {
    GetOutput() = {};
    // return false;  // Или true, если хотите просто пропустить
    return true;
  } else {
    const std::vector<int> &data = input.data;
    size_t data_size = data.size();
    // auto data_size = static_cast<uint64_t>(data.size());
    GetOutput().resize(data_size);

    if (source == dest) {
      HandleSameSourceDestination(rank, source, data_size, input.data, GetOutput());
    } else {
      HandleDifferentSourceDestination(rank, source, dest, data_size, input.data, GetOutput());
    }
  }
  return true;
}

bool BarkalovaMStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_star
