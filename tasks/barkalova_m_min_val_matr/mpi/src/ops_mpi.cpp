#include "barkalova_m_min_val_matr/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <utility>
#include <vector>

#include "barkalova_m_min_val_matr/common/include/common.hpp"

namespace barkalova_m_min_val_matr {

BarkalovaMMinValMatrMPI::BarkalovaMMinValMatrMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput().clear();
}

bool BarkalovaMMinValMatrMPI::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return false;
  }
  size_t stolb = matrix[0].size();
  return std::ranges::all_of(matrix, [stolb](const auto &row) { return row.size() == stolb; });
}

bool BarkalovaMMinValMatrMPI::PreProcessingImpl() {
  if (!GetInput().empty()) {
    size_t stolb = GetInput()[0].size();
    GetOutput().resize(stolb, INT_MAX);
  } else {
    GetOutput().clear();
  }
  return true;
}

bool BarkalovaMMinValMatrMPI::RunImpl() {
  const auto &matrix = GetInput();
  auto &res = GetOutput();

  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  size_t rows = matrix.size();
  size_t stolb = matrix[0].size();
  // Распределение столбцов
  size_t loc_stolb = stolb / size;
  size_t ostatok = stolb % size;
  size_t start_stolb = (rank * loc_stolb) + (std::cmp_less(rank, ostatok) ? rank : ostatok);
  size_t end_stolb = start_stolb + loc_stolb + (std::cmp_less(rank, ostatok) ? 1 : 0);
  size_t col_stolb = end_stolb - start_stolb;

  std::vector<int> loc_min(col_stolb, INT_MAX);
  for (size_t k = 0; k < col_stolb; ++k) {
    size_t stolb_index = start_stolb + k;
    for (size_t i = 0; i < rows; ++i) {
      loc_min[k] = std::min(matrix[i][stolb_index], loc_min[k]);
    }
  }

  std::vector<int> recv_counts(size);
  std::vector<int> displacements(size);

  for (int i = 0; i < size; i++) {
    size_t i_start = (i * loc_stolb) + (std::cmp_less(i, ostatok) ? i : ostatok);
    size_t i_end = i_start + loc_stolb + (std::cmp_less(i, ostatok) ? 1 : 0);
    recv_counts[i] = static_cast<int>(i_end - i_start);
    displacements[i] = static_cast<int>(i_start);
  }
  res.resize(stolb, INT_MAX);

  int send_count = static_cast<int>(col_stolb);

  MPI_Gatherv(loc_min.data(), send_count, MPI_INT, res.data(), recv_counts.data(), displacements.data(), MPI_INT, 0,
              MPI_COMM_WORLD);

  if (size > 1) {
    int stolb_int = static_cast<int>(stolb);
    MPI_Bcast(res.data(), stolb_int, MPI_INT, 0, MPI_COMM_WORLD);
  }

  return true;
}

bool BarkalovaMMinValMatrMPI::PostProcessingImpl() {
  return true;
}
}  // namespace barkalova_m_min_val_matr
