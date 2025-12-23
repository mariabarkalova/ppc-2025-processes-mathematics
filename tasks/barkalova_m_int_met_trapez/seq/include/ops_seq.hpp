#pragma once

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
#include "task/include/task.hpp"

namespace barkalova_m_int_met_trapez {

class BarkalovaMIntMetTrapezSEQ : public BaseTask {
 public:
  static constexpr ppc::task::TypeOfTask GetStaticTypeOfTask() {
    return ppc::task::TypeOfTask::kSEQ;
  }
  explicit BarkalovaMIntMetTrapezSEQ(const InType &in);

 private:
  bool ValidationImpl() override;
  bool PreProcessingImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;
};

}  // namespace barkalova_m_int_met_trapez
