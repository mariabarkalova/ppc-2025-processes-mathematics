/*#include "barkalova_m_min_val_matr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "barkalova_m_min_val_matr/common/include/common.hpp"

namespace barkalova_m_min_val_matr {

BarkalovaMMinValMatrSEQ::BarkalovaMMinValMatrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    GetInput()[i] = in[i];
  }
  GetOutput().clear();
}

bool BarkalovaMMinValMatrSEQ::ValidationImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  size_t stolb = matrix[0].size();
  // все строки должны иметь одинак разм
  return std::ranges::all_of(matrix, [stolb](const auto &row) { return row.size() == stolb; });
}

bool BarkalovaMMinValMatrSEQ::PreProcessingImpl() {
  if (!GetInput().empty()) {
    size_t stolb = GetInput()[0].size();
    GetOutput().resize(stolb, INT_MAX);
  } else {
    GetOutput().clear();
  }
  return true;
}

bool BarkalovaMMinValMatrSEQ::RunImpl() {
  const auto &matrix = GetInput();
  auto &res = GetOutput();
  if (matrix.empty()) {
    return true;
  }
  for (size_t j = 0; j < res.size(); ++j) {
    int min_val = INT_MAX;
    for (const auto &row : matrix) {
      min_val = std::min(row[j], min_val);
    }
    res[j] = min_val;
  }

  return true;
}

bool BarkalovaMMinValMatrSEQ::PostProcessingImpl() {
  return GetInput().empty() || !GetOutput().empty();
}
}  // namespace barkalova_m_min_val_matr
*/

#include "barkalova_m_min_val_matr/seq/include/ops_seq.hpp"

#include <algorithm>
#include <climits>
#include <cstddef>
#include <vector>

#include "barkalova_m_min_val_matr/common/include/common.hpp"

namespace barkalova_m_min_val_matr {

BarkalovaMMinValMatrSEQ::BarkalovaMMinValMatrSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput().clear();
  GetInput().reserve(in.size());
  for (const auto &row : in) {
    GetInput().push_back(row);
  }

  GetOutput().clear();
}

bool BarkalovaMMinValMatrSEQ::ValidationImpl() {
  const auto &input = GetInput();
  if (input.empty()) {
    return true;
  }

  size_t length_row = input[0].size();
  return std::ranges::all_of(input, [length_row](const auto &row) { return row.size() == length_row; });
}

bool BarkalovaMMinValMatrSEQ::PreProcessingImpl() {
  if (GetInput().empty()) {
    return true;
  }
  GetOutput().resize(GetInput()[0].size(), INT_MAX);
  return true;
}

bool BarkalovaMMinValMatrSEQ::RunImpl() {
  const auto &matrix = GetInput();
  if (matrix.empty()) {
    return true;
  }
  auto &output = GetOutput();

  for (size_t i = 0; i < output.size(); i++) {
    for (const auto &row : matrix) {
      output[i] = std::min(output[i], row[i]);
    }
  }

  return true;
}

bool BarkalovaMMinValMatrSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_min_val_matr
