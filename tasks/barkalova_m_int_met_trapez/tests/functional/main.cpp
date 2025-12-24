#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
#include "barkalova_m_int_met_trapez/mpi/include/ops_mpi.hpp"
#include "barkalova_m_int_met_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_int_met_trapez {

struct TestIntegralData {
  std::vector<int> n_i;
  std::vector<std::pair<double, double>> limits;
  double expected_value;
  double tolerance;
};

using FuncTestType = std::tuple<TestIntegralData, std::string>;

class BarkalovaMIntMetTrapezRunFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, FuncTestType> {
 public:
  static std::string PrintTestParam(const FuncTestType &test_param) {
    const auto &data = std::get<0>(test_param);
    std::string desc = std::get<1>(test_param);
    return desc + "_nx" + std::to_string(data.n_i[0]) + "_ny" + std::to_string(data.n_i[1]);
  }

 protected:
  void SetUp() override {
    FuncTestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = std::get<0>(params);

    Integral integral_data;
    integral_data.limits = input_data_.limits;
    integral_data.n_i = input_data_.n_i;

    test_input_ = integral_data;
    expected_value_ = input_data_.expected_value;
    tolerance_ = input_data_.tolerance;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double absolute_error = std::abs(output_data - expected_value_);

    if (absolute_error >= tolerance_) {
      std::cout << "TEST FAILED: absolute error = " << absolute_error << ", tolerance = " << tolerance_
                << ", expected = " << expected_value_ << ", got = " << output_data << '\n';
    }
    return absolute_error < tolerance_;
  }

  InType GetTestInputData() final {
    return test_input_;
  }

 private:
  TestIntegralData input_data_{};
  InType test_input_{};
  double expected_value_ = 0.0;
  double tolerance_ = 1e-6;
};

namespace {

TEST_P(BarkalovaMIntMetTrapezRunFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

const std::array<FuncTestType, 8> kTestParam = {
    // Единичный квадрат [0,1]x[0,1]
    std::make_tuple(TestIntegralData{.n_i = {100, 100},
                                     .limits = {{0.0, 1.0}, {0.0, 1.0}},
                                     .expected_value = 2.0 / 3.0,  // 0.666667
                                     .tolerance = 1e-4},
                    "unit_square_100x100"),

    // Тот же квадрат с меньшей сеткой
    std::make_tuple(TestIntegralData{.n_i = {50, 50},
                                     .limits = {{0.0, 1.0}, {0.0, 1.0}},
                                     .expected_value = 2.0 / 3.0,  // 0.666667
                                     .tolerance = 1e-3},
                    "unit_square_50x50"),

    // Прямоугольник [0,2]x[0,3]
    std::make_tuple(TestIntegralData{.n_i = {100, 100},
                                     .limits = {{0.0, 2.0}, {0.0, 3.0}},
                                     .expected_value = 26.0,  // (8*3 + 27*2)/3 = 78/3 = 26
                                     .tolerance = 1e-2},
                    "rectangle_2x3_100x100"),

    // Симметричный квадрат [-1,1]x[-1,1]
    std::make_tuple(TestIntegralData{.n_i = {100, 100},
                                     .limits = {{-1.0, 1.0}, {-1.0, 1.0}},
                                     .expected_value = 8.0 / 3.0,  // 2.666667
                                     .tolerance = 1e-3},
                    "symmetric_square_100x100"),

    // Прямоугольник [0,2]x[0,1]
    std::make_tuple(TestIntegralData{.n_i = {200, 50},
                                     .limits = {{0.0, 2.0}, {0.0, 1.0}},
                                     .expected_value = 10.0 / 3.0,  // 3.333333
                                     .tolerance = 1e-3},
                    "nonuniform_grid_200x50"),

    // Маленькая сетка [0,1]x[0,1]
    std::make_tuple(TestIntegralData{.n_i = {10, 10},
                                     .limits = {{0.0, 1.0}, {0.0, 1.0}},
                                     .expected_value = 2.0 / 3.0,  // 0.666667
                                     .tolerance = 0.1},
                    "small_grid_10x10"),

    // Минимальная сетка [0,1]x[0,1]
    std::make_tuple(TestIntegralData{.n_i = {1, 1},
                                     .limits = {{0.0, 1.0}, {0.0, 1.0}},
                                     .expected_value = 2.0 / 3.0,  // 0.666667
                                     .tolerance = 0.4},
                    "minimal_grid_1x1"),

    // Большая сетка для точности [0,1]x[0,1]
    std::make_tuple(TestIntegralData{.n_i = {500, 500},
                                     .limits = {{0.0, 1.0}, {0.0, 1.0}},
                                     .expected_value = 2.0 / 3.0,  // 0.666667
                                     .tolerance = 1e-5},
                    "high_precision_500x500")};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BarkalovaMIntMetTrapezMPI, InType>(kTestParam, PPC_SETTINGS_barkalova_m_int_met_trapez),
    ppc::util::AddFuncTask<BarkalovaMIntMetTrapezSEQ, InType>(kTestParam, PPC_SETTINGS_barkalova_m_int_met_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BarkalovaMIntMetTrapezRunFuncTests::PrintFuncTestName<BarkalovaMIntMetTrapezRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegrationTests, BarkalovaMIntMetTrapezRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace barkalova_m_int_met_trapez
