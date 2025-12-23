#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace barkalova_m_star {
struct StarMessage {
  int source{};  // Узел-отправитель
  int dest{};    // Узел-получатель
  std::vector<int> data;
};

using InType = StarMessage;
using OutType = std::vector<int>;
using TestType = std::tuple<StarMessage, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace barkalova_m_star
