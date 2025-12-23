#pragma once

#include <string>
#include <tuple>

#include "task/include/task.hpp"

namespace barkalova_m_int_met_trapez {

struct Integral {
  double function(double x, double y)  // подинтегральная функция
  {
    return x * x + y * y;
  }

  std::vector<std::pair<double, double>> limits;  // пределы интегрирования
  std::vector<int> n_i;                           // число узлов интегрирования
};
using InType = Integral;
using OutType = double;
using TestType = std::tuple<int, std::string>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace barkalova_m_int_met_trapez
