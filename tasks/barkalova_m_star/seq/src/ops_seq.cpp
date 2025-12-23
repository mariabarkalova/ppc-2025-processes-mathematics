#include "barkalova_m_star/seq/include/ops_seq.hpp"

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
