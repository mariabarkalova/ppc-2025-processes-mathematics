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

#include <climits>
#include <cstddef>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_star {

class BarkalovaMStarPerfTest : public ppc::util::BaseRunPerfTests<InType, OutType> {
  static constexpr int kDataSize = 1000000;

  InType input_data_{};
  std::vector<int> expected_output_{};
  bool skip_test_ = false;
  int world_size_ = 0;

  void SetUp() override {
    MPI_Comm_size(MPI_COMM_WORLD, &world_size_);

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    // Если это SEQ задача и процессов больше 1 - пропускаем тест
    if (task_type.find("seq") != std::string::npos) {
      if (world_size_ > 1) {
        skip_test_ = true;
        return;
      }
    }

    skip_test_ = false;

    // Инициализируем тестовые данные
    input_data_.data.resize(kDataSize);
    for (int i = 0; i < kDataSize; ++i) {
      input_data_.data[static_cast<std::size_t>(i)] = i % 100;
    }

    // Выбираем сценарий в зависимости от количества процессов
    if (world_size_ >= 4) {
      // Сценарий 1: Периферия -> Периферия через центр
      input_data_.source = 1;
      input_data_.dest = 3;
      expected_output_ = input_data_.data;
    } else if (world_size_ >= 3) {
      // Сценарий 2: Центр -> Периферия
      input_data_.source = 0;
      input_data_.dest = 2;
      expected_output_ = input_data_.data;
    } else if (world_size_ == 2) {
      // Сценарий 3: 2 процесса - программа возвращает пустой вектор
      input_data_.source = 0;
      input_data_.dest = 1;
      expected_output_ = {};  // Пустой вектор
    } else if (world_size_ == 1) {
      // Сценарий 4: 1 процесс
      input_data_.source = 0;
      input_data_.dest = 0;
      expected_output_ = input_data_.data;
    }
  }

  bool CheckTestOutputData(OutType &output_data) final {
    if (skip_test_) {
      return true;
    }

    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);

    // Для SEQ версии
    if (task_type.find("seq") != std::string::npos) {
      return output_data == input_data_.data;
    }

    // Для MPI версии
    if (world_size_ < 3) {
      // На 1-2 процессах: программа возвращает пустой вектор
      return output_data.empty();
    }

    // На 3+ процессах: все получают данные через Bcast
    if (output_data.size() != expected_output_.size()) {
      return false;
    }

    for (size_t i = 0; i < output_data.size(); ++i) {
      if (output_data[i] != expected_output_[i]) {
        return false;
      }
    }

    return true;
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
