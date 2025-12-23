#pragma once

#include "barkalova_m_star/common/include/common.hpp"
#include "task/include/task.hpp"

namespace barkalova_m_star {

class BarkalovaMStarSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BarkalovaMStarSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace barkalova_m_star
