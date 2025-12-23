/*#include <gtest/gtest.h>
#include <stb/stb_image.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
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

class NesterovARunFuncTestsProcesses3 : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
 public:
  static std::string PrintTestParam(const TestType &test_param) {
    return std::to_string(std::get<0>(test_param)) + "_" + std::get<1>(test_param);
  }

 protected:
  void SetUp() override {
    int width = -1;
    int height = -1;
    int channels = -1;
    std::vector<uint8_t> img;
    // Read image
    {
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_example_processes_3, "pic.jpg");
      auto *data = stbi_load(abs_path.c_str(), &width, &height, &channels, 0);
      if (data == nullptr) {
        throw std::runtime_error("Failed to load image: " + std::string(stbi_failure_reason()));
      }
      img = std::vector<uint8_t>(data, data + (static_cast<ptrdiff_t>(width * height * channels)));
      stbi_image_free(data);
      if (std::cmp_not_equal(width, height)) {
        throw std::runtime_error("width != height: ");
      }
    }

    TestType params = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());
    input_data_ = width - height + std::min(std::accumulate(img.begin(), img.end(), 0), channels);
  }

  bool CheckTestOutputData(OutType &output_data) final {
    return (input_data_ == output_data);
  }

  InType GetTestInputData() final {
    return input_data_;
  }

 private:
  InType input_data_ = 0;
};

namespace {

TEST_P(NesterovARunFuncTestsProcesses3, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BarkalovaMIntMetTrapezMPI, InType>(kTestParam,
PPC_SETTINGS_example_processes_3), ppc::util::AddFuncTask<BarkalovaMIntMetTrapezSEQ, InType>(kTestParam,
PPC_SETTINGS_example_processes_3));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = NesterovARunFuncTestsProcesses3::PrintFuncTestName<NesterovARunFuncTestsProcesses3>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, NesterovARunFuncTestsProcesses3, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace barkalova_m_int_met_trapez
*/

/*
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <tuple>

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
#include "barkalova_m_int_met_trapez/mpi/include/ops_mpi.hpp"
#include "barkalova_m_int_met_trapez/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_int_met_trapez {

// Структура для тестовых данных
struct TestIntegralData {
  std::vector<int> n_i;  // Число интервалов по x и y
  std::vector<std::pair<double, double>> limits;  // Пределы интегрирования
  double expected_value;  // Ожидаемое значение интеграла
  double tolerance;       // Допустимая погрешность
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

    // Создаем входные данные для структуры Integral
    Integral integral_data;
    integral_data.limits = input_data_.limits;
    integral_data.n_i = input_data_.n_i;

    test_input_ = integral_data;
    expected_value_ = input_data_.expected_value;
    tolerance_ = input_data_.tolerance;
  }

  bool CheckTestOutputData(OutType &output_data) final {
    double error = std::abs(output_data - expected_value_);

    // Для отладки выводим информацию при ошибке
    if (error >= tolerance_) {
      std::cout << "TEST FAILED: error = " << error
                << ", tolerance = " << tolerance_
                << ", expected = " << expected_value_
                << ", got = " << output_data << std::endl;
    }

    return error < tolerance_;
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

// ПРАВИЛЬНАЯ аналитическая формула для f(x,y) = x² + y²
double CalculateExactIntegral(double x1, double x2, double y1, double y2) {
  // ∫∫(x² + y²)dxdy = ∫(x²y + y³/3)|_{y1}^{y2} dx от x1 до x2
  // = ∫[x²(y2-y1) + (y2³ - y1³)/3]dx от x1 до x2
  // = (y2-y1)(x2³ - x1³)/3 + (y2³ - y1³)(x2 - x1)/3

  double x_diff = x2 - x1;
  double y_diff = y2 - y1;

  double x_cubed_diff = (x2*x2*x2 - x1*x1*x1);
  double y_cubed_diff = (y2*y2*y2 - y1*y1*y1);

  return (x_cubed_diff * y_diff + y_cubed_diff * x_diff) / 3.0;
}

TEST_P(BarkalovaMIntMetTrapezRunFuncTests, IntegrationTest) {
  ExecuteTest(GetParam());
}

// Упрощенные тесты с ПРАВИЛЬНЫМИ аналитическими значениями
const std::array<FuncTestType, 8> kTestParam = {
    // Тест 1: Единичный квадрат [0,1]x[0,1]
    // ∫∫(x²+y²)dxdy = ∫(x²y + y³/3)|_{0}^{1} dx от 0 до 1
    // = ∫(x² + 1/3)dx от 0 до 1 = (1/3 + 1/3) = 2/3 = 0.6666667
    std::make_tuple(
        TestIntegralData{
            .n_i = {100, 100},
            .limits = {{0.0, 1.0}, {0.0, 1.0}},
            .expected_value = 2.0/3.0,  // 0.6666667
            .tolerance = 1e-4
        },
        "unit_square_100x100"
    ),

    // Тест 2: Тот же квадрат с меньшей сеткой
    std::make_tuple(
        TestIntegralData{
            .n_i = {50, 50},
            .limits = {{0.0, 1.0}, {0.0, 1.0}},
            .expected_value = 2.0/3.0,
            .tolerance = 1e-3  // Больший допуск для меньшей сетки
        },
        "unit_square_50x50"
    ),

    // Тест 3: Прямоугольник [0,2]x[0,3]
    // ∫∫(x²+y²)dxdy = (8*3 + 27*2)/3 = (24 + 54)/3 = 78/3 = 26
    std::make_tuple(
        TestIntegralData{
            .n_i = {100, 100},
            .limits = {{0.0, 2.0}, {0.0, 3.0}},
            .expected_value = 26.0,
            .tolerance = 1e-2
        },
        "rectangle_2x3_100x100"
    ),

    // Тест 4: Симметричный квадрат [-1,1]x[-1,1]
    // Можно вычислить как 4 * ∫∫(x²+y²)dxdy от 0 до 1
    // = 4 * 2/3 = 8/3 = 2.6666667
    std::make_tuple(
        TestIntegralData{
            .n_i = {100, 100},
            .limits = {{-1.0, 1.0}, {-1.0, 1.0}},
            .expected_value = 8.0/3.0,  // 2.6666667
            .tolerance = 1e-3
        },
        "symmetric_square_100x100"
    ),

    // Тест 5: Прямоугольник [0,2]x[0,1]
    // ∫∫(x²+y²)dxdy = (8*1 + 1*2)/3 = (8 + 2)/3 = 10/3 = 3.333333
    std::make_tuple(
        TestIntegralData{
            .n_i = {200, 50},
            .limits = {{0.0, 2.0}, {0.0, 1.0}},
            .expected_value = 10.0/3.0,  // 3.333333
            .tolerance = 1e-3
        },
        "nonuniform_grid_200x50"
    ),

    // Тест 6: Маленькая сетка
    std::make_tuple(
        TestIntegralData{
            .n_i = {10, 10},
            .limits = {{0.0, 1.0}, {0.0, 1.0}},
            .expected_value = 2.0/3.0,
            .tolerance = 1e-1  // Очень большой допуск для маленькой сетки
        },
        "small_grid_10x10"
    ),

    // Тест 7: Минимальная сетка (1 интервал = 2 узла)
    // Метод трапеций для f(x,y)=x²+y² на [0,1]x[0,1] с 2x2 узлами:
    // Узлы: (0,0), (0,1), (1,0), (1,1)
    // Значения: 0, 1, 1, 2
    // Веса: все углы = 0.25
    // Сумма: (0+1+1+2)*0.25 = 4*0.25 = 1.0
    // hx=hy=1, результат: 1.0*1*1 = 1.0
    // Но точное значение = 0.6667, погрешность большая
    std::make_tuple(
        TestIntegralData{
            .n_i = {1, 1},
            .limits = {{0.0, 1.0}, {0.0, 1.0}},
            .expected_value = 2.0/3.0,
            .tolerance = 0.4  // Очень большой допуск
        },
        "minimal_grid_1x1"
    ),

    // Тест 8: Большая сетка для точного вычисления
    std::make_tuple(
        TestIntegralData{
            .n_i = {500, 500},
            .limits = {{0.0, 1.0}, {0.0, 1.0}},
            .expected_value = 2.0/3.0,
            .tolerance = 1e-5
        },
        "high_precision_500x500"
    )
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BarkalovaMIntMetTrapezMPI, InType>(
        kTestParam, PPC_SETTINGS_barkalova_m_int_met_trapez),
    ppc::util::AddFuncTask<BarkalovaMIntMetTrapezSEQ, InType>(
        kTestParam, PPC_SETTINGS_barkalova_m_int_met_trapez));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName =
    BarkalovaMIntMetTrapezRunFuncTests::PrintFuncTestName<BarkalovaMIntMetTrapezRunFuncTests>;

INSTANTIATE_TEST_SUITE_P(IntegrationTests, BarkalovaMIntMetTrapezRunFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace barkalova_m_int_met_trapez

*/
