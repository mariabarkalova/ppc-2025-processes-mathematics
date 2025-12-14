/*
#include "barkalova_m_star/seq/include/ops_seq.hpp"

#include <numeric>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_star {

BarkalovaMStarSEQ::BarkalovaMStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BarkalovaMStarSEQ::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool BarkalovaMStarSEQ::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool BarkalovaMStarSEQ::RunImpl() {
  if (GetInput() == 0) {
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

  int counter = 0;
  for (int i = 0; i < num_threads; i++) {
    counter++;
  }

  if (counter != 0) {
    GetOutput() /= counter;
  }
  return GetOutput() > 0;
}

bool BarkalovaMStarSEQ::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace barkalova_m_star
*/

#include "barkalova_m_star/seq/include/ops_seq.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"

namespace barkalova_m_star {

BarkalovaMStarSEQ::BarkalovaMStarSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BarkalovaMStarSEQ::ValidationImpl() {
  const auto &input = GetInput();

  // В SEQ всегда 1 процесс (rank = 0, size = 1)
  int size = 1;
  if (input.source < 0 || input.source >= size) {
    return false;
  }
  if (input.dest < 0 || input.dest >= size) {
    return false;
  }
  return true;
}

bool BarkalovaMStarSEQ::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool BarkalovaMStarSEQ::RunImpl() {
  GetOutput() = GetInput().data;
  return true;
}

bool BarkalovaMStarSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_star
