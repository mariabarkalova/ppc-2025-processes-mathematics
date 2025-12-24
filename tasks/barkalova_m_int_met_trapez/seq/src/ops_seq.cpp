#include "barkalova_m_int_met_trapez/seq/include/ops_seq.hpp"

#include <vector>

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
// #include "util/include/util.hpp"

namespace barkalova_m_int_met_trapez {

BarkalovaMIntMetTrapezSEQ::BarkalovaMIntMetTrapezSEQ(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BarkalovaMIntMetTrapezSEQ::ValidationImpl() {
  auto &data = GetInput();
  return data.limits.size() >= 2 && data.n_i.size() >= 2;
}

bool BarkalovaMIntMetTrapezSEQ::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}

bool BarkalovaMIntMetTrapezSEQ::RunImpl() {
  auto data = GetInput();

  if (data.n_i.size() < 2 || data.limits.size() < 2) {
    GetOutput() = 0.0;
    return true;
  }

  double x1 = data.limits[0].first;
  double x2 = data.limits[0].second;
  double y1 = data.limits[1].first;
  double y2 = data.limits[1].second;
  int n_steps_x = data.n_i[0];
  int n_steps_y = data.n_i[1];
  double hx = (x2 - x1) / n_steps_x;
  double hy = (y2 - y1) / n_steps_y;

  double sum = 0.0;

  for (int i = 0; i <= n_steps_x; ++i) {
    double x = x1 + (i * hx);
    double weight_x = (i == 0 || i == n_steps_x) ? 0.5 : 1.0;

    for (int j = 0; j <= n_steps_y; ++j) {
      double y = y1 + (j * hy);
      double weight_y = (j == 0 || j == n_steps_y) ? 0.5 : 1.0;
      sum += Integral::Function(x, y) * weight_x * weight_y;
    }
  }
  GetOutput() = sum * hx * hy;
  return true;
}

bool BarkalovaMIntMetTrapezSEQ::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_int_met_trapez
