/*
#include <gtest/gtest.h>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_star {

class BarkalovaMStarPerfTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
  const int kCount_ = 100;
  InType input_data_{};

  void SetUp() override {
    input_data_ = kCount_;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return input_data_ == output_data;
  }

  InType GetTestInputData() final {
    return input_data_;
  }
};

TEST_P(BarkalovaMStarPerfTests, RunPerfModes) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks =
    ppc::util::MakeAllPerfTasks<InType, BarkalovaMStarMPI, BarkalovaMStarSEQ>(PPC_SETTINGS_barkalova_m_star);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);

const auto kPerfTestName = BarkalovaMStarPerfTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(RunModeTests, BarkalovaMStarPerfTests, kGtestValues, kPerfTestName);

}  // namespace barkalova_m_star
*/

#include <gtest/gtest.h>
#include <mpi.h>

#include <cstddef>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_star {

class BarkalovaMStarPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 10000000;
  InType input_data_{};
  bool skip_test_ = false;

  void SetUp() override {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    // Если это SEQ задача и процессов больше 1 - пропускаем тест
    if (task_type.find("seq") != std::string::npos) {
      if (size > 1) {
        skip_test_ = true;
        return;
      }
    }

    skip_test_ = false;

    if (size >= 4) {
      // На 4+ процессах
      input_data_.source = 1;  // периферия
      input_data_.dest = 3;    // периферия
    } else if (size >= 3) {
      // На 3 процессах
      input_data_.source = 0;  // центр
      input_data_.dest = 2;    // периферия
    } else if (size >= 2) {
      // На 2 процессах
      input_data_.source = 0;  // центр
      input_data_.dest = 1;    // периферия
    } else {
      // На 1 процессе: SEQ версия или широковещание
      input_data_.source = 0;  // центр
      input_data_.dest = 0;    // широковещание
    }

    input_data_.data.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[static_cast<std::size_t>(i)] = i % 100;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (skip_test_) {
      return true;
    }

    int rank = 0;
    int size = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    if (task_type.find("seq") != std::string::npos) {
      return output_data == input_data_.data;
    }

    if (input_data_.source == 0 && input_data_.dest == 0) {
      return output_data == input_data_.data;
    } else if (rank == input_data_.dest) {
      return output_data == input_data_.data;
    } else {
      return output_data.empty();
    }
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
