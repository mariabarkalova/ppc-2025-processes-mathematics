#pragma once

#include <tuple>
#include <vector>

#include "task/include/task.hpp"

namespace barkalova_m_star {
struct StarMessage {
  int source;  // Узел-отправитель (может быть не только центр!)
  int dest;    // Узел-получатель
  std::vector<int> data;
};

using InType = StarMessage;
// using InType = std::tuple<int, int, std::vector<int>>;
using OutType = std::vector<int>;
using TestType = std::tuple<StarMessage, std::vector<int>>;
using BaseTask = ppc::task::Task<InType, OutType>;
// Измените TestType на tuple<int, int, int, int> для нашего шаблона
// (center, source, dest, data_size)
// using TestType = std::tuple<int, int, int, int>;
// using BaseTask = ppc::task::Task<InType, OutType>;

}  // namespace barkalova_m_star
