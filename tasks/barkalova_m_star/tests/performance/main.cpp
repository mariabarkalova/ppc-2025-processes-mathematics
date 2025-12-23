#include <gtest/gtest.h>
#include <mpi.h>

#include <climits>
#include <cstddef>
#include <string>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_star {

class BarkalovaMStarPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 10000000;

  InType input_data_{};
  std::vector<int> expected_output_;
  bool skip_test_ = false;
  int world_size_ = 0;
  int world_rank_ = 0;

  void SetUp() override {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank_);

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    if (task_type.find("seq") != std::string::npos && world_size_ > 1) {
      skip_test_ = true;
      return;
    }

    skip_test_ = false;

    input_data_.data.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[static_cast<std::size_t>(i)] = i % 100;
    }

    if (world_size_ >= 4) {
      input_data_.source = 1;
      input_data_.dest = 3;
    } else if (world_size_ >= 3) {
      input_data_.source = 0;
      input_data_.dest = 2;
    } else if (world_size_ == 2) {
      input_data_.source = 0;
      input_data_.dest = 1;
    } else {
      input_data_.source = 0;
      input_data_.dest = 0;
    }

    if (input_data_.source == input_data_.dest) {
      expected_output_ = input_data_.data;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (skip_test_) {
      return true;
    }

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    if (task_type.find("seq") != std::string::npos) {
      return output_data == input_data_.data;
    }

    if (input_data_.source == input_data_.dest) {
      return output_data == input_data_.data;
    }

    if (world_rank_ == input_data_.dest) {
      return output_data == input_data_.data;
    }

    return output_data.empty();
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BarkalovaMStarPerfTest, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BarkalovaMStarMPI, BarkalovaMStarSEQ>(PPC_SETTINGS_barkalova_m_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BarkalovaMStarPerfTest::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BarkalovaMStarPerfTest, kGtestValues, kPerfTestName);

}  // namespace barkalova_m_star
