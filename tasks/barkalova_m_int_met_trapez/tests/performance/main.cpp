#include <gtest/gtest.h>
#include <mpi.h>

#include <cmath>
#include <iostream>

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
#include "barkalova_m_int_met_trapez/mpi/include/ops_mpi.hpp"
#include "barkalova_m_int_met_trapez/seq/include/ops_seq.hpp"
#include "util/include/perf_test_util.hpp"

namespace barkalova_m_int_met_trapez {

class BarkalovaIntegralPerformanceTests : public ppc::util::BaseRunPerfTests<InType, OutType> {
 protected:
  void SetUp() override {
    input_data_ = Integral{
        .limits = {{0.0, 1.0}, {0.0, 1.0}},  // Единичный квадрат
        .n_i = {500, 500}                    //  сетка для теста
    };
  }

  bool CheckTestOutputData(OutType &output_data) final {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank != 0) {
      return true;
    }

    double expected = 2.0 / 3.0;
    double tolerance = 1e-4;

    double error = std::abs(output_data - expected);
    bool result = error < tolerance;

    if (!result) {
      std::cout << "Performance test check failed: expected " << expected << ", got " << output_data << ", error "
                << error << ", tolerance " << tolerance << '\n';
    }

    return result;
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_{};
};

TEST_P(BarkalovaIntegralPerformanceTests, PerformanceTests) {
  ExecuteTest(GetParam());
}

const auto kAllPerfTasks = ppc::util::MakeAllPerfTasks<InType, BarkalovaMIntMetTrapezMPI, BarkalovaMIntMetTrapezSEQ>(
    PPC_SETTINGS_barkalova_m_int_met_trapez);

const auto kGtestValues = ppc::util::TupleToGTestValues(kAllPerfTasks);
const auto kPerfTestName = BarkalovaIntegralPerformanceTests::CustomPerfTestName;

INSTANTIATE_TEST_SUITE_P(BarkalovaPerformanceTestSuite, BarkalovaIntegralPerformanceTests, kGtestValues, kPerfTestName);

}  // namespace barkalova_m_int_met_trapez
