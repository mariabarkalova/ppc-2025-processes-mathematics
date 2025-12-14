/*
#include <gtest/gtest.h>
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

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_star {

class BarkalovaMStarFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestType> {
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
      std::string abs_path = ppc::util::GetAbsoluteTaskPath(PPC_ID_barkalova_m_star, "pic.jpg");
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

TEST_P(BarkalovaMStarFuncTests, MatmulFromPic) {
  ExecuteTest(GetParam());
}

const std::array<TestType, 3> kTestParam = {std::make_tuple(3, "3"), std::make_tuple(5, "5"), std::make_tuple(7, "7")};

const auto kTestTasksList =
    std::tuple_cat(ppc::util::AddFuncTask<BarkalovaMStarMPI, InType>(kTestParam, PPC_SETTINGS_barkalova_m_star),
                   ppc::util::AddFuncTask<BarkalovaMStarSEQ, InType>(kTestParam, PPC_SETTINGS_barkalova_m_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BarkalovaMStarFuncTests::PrintFuncTestName<BarkalovaMStarFuncTests>;

INSTANTIATE_TEST_SUITE_P(PicMatrixTests, BarkalovaMStarFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace barkalova_m_star
*/

#include <gtest/gtest.h>
#include <mpi.h>

#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"

namespace barkalova_m_star {

class BarkalovaMStarFuncTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    int is_mpi_initialized;
    MPI_Initialized(&is_mpi_initialized);
    if (!is_mpi_initialized) {
      MPI_Init(nullptr, nullptr);
    }
  }

  static void TearDownTestSuite() {
    int is_mpi_initialized;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized) {
      int is_mpi_finalized;
      MPI_Finalized(&is_mpi_finalized);
      if (!is_mpi_finalized) {
        MPI_Finalize();
      }
    }
  }

  static int GetWorldSize() {
    int size = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
  }

  static int GetWorldRank() {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
  }
};

// Тест 1: Широковещание от центра всем
TEST_F(BarkalovaMStarFuncTest, BroadcastFromCenter) {
  int size = GetWorldSize();
  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();

  // Только процесс 0 инициализирует данные!
  if (rank == 0) {
    input.source = 0;
    input.dest = 0;
    input.data = {1, 2, 3, 4, 5};
  } else {
    input.source = 0;
    input.dest = 0;
    input.data = {};  // Пустые данные на остальных процессах
  }

  BarkalovaMStarMPI task(input);

  // Только процесс 0 должен проходить валидацию с данными
  if (rank == 0) {
    ASSERT_TRUE(task.Validation());
  } else {
    // На других процессах валидация должна пройти с пустыми данными
    ASSERT_TRUE(task.Validation());
  }

  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // Все процессы должны получить данные
  auto output = task.GetOutput();

  // Проверяем что все получили правильные данные
  if (rank == 0) {
    EXPECT_EQ(output, std::vector<int>({1, 2, 3, 4, 5}));
  } else {
    // Остальные тоже должны получить данные через broadcast
    EXPECT_EQ(output, std::vector<int>({1, 2, 3, 4, 5}));
  }
}

// Тест 2: Отправка от центра к периферийному узлу
TEST_F(BarkalovaMStarFuncTest, SendFromCenterToPeripheral) {
  int size = GetWorldSize();
  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();

  // Только процесс 0 инициализирует данные!
  if (rank == 0) {
    input.source = 0;
    input.dest = 2;
    input.data = {10, 20, 30, 40};
  } else {
    input.source = 0;
    input.dest = 2;
    input.data = {};  // Пустые данные на остальных процессах
  }

  BarkalovaMStarMPI task(input);

  // Проверяем валидацию
  if (rank == 0) {
    ASSERT_TRUE(task.Validation());
  } else {
    ASSERT_TRUE(task.Validation());  // Должна пройти с пустыми данными
  }

  ASSERT_TRUE(task.PreProcessing());
  bool run_result = task.Run();
  ASSERT_TRUE(run_result);
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 2) {
    // Только получатель должен иметь данные
    EXPECT_EQ(output, std::vector<int>({10, 20, 30, 40}));
  } else {
    // Центр и остальные процессы НЕ должны иметь данные в выходе
    EXPECT_TRUE(output.empty());
  }
}

// Тест 3: Отправка от периферии к центру
TEST_F(BarkalovaMStarFuncTest, SendFromPeripheralToCenter) {
  int size = GetWorldSize();
  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();

  // Только процесс 0 инициализирует данные!
  if (rank == 0) {
    input.source = 1;
    input.dest = 0;
    input.data = {100, 200, 300};
  } else {
    input.source = 1;
    input.dest = 0;
    input.data = {};  // Пустые данные на остальных процессах
  }

  BarkalovaMStarMPI task(input);

  // Проверяем валидацию
  if (rank == 0) {
    ASSERT_TRUE(task.Validation());
  } else {
    ASSERT_TRUE(task.Validation());
  }

  ASSERT_TRUE(task.PreProcessing());
  bool run_result = task.Run();
  ASSERT_TRUE(run_result);
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 0) {
    // Только центр (получатель) должен иметь данные
    EXPECT_EQ(output, std::vector<int>({100, 200, 300}));
  } else {
    // Отправитель и остальные процессы НЕ должны иметь данные
    EXPECT_TRUE(output.empty());
  }
}

// Тест 4: Отправка между периферийными узлами через центр
TEST_F(BarkalovaMStarFuncTest, SendBetweenPeripheralsThroughCenter) {
  int size = GetWorldSize();
  if (size < 4) {
    GTEST_SKIP() << "Need at least 4 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();

  // Только процесс 0 инициализирует данные!
  if (rank == 0) {
    input.source = 1;
    input.dest = 3;
    input.data = {5, 10, 15, 20, 25};
  } else {
    input.source = 1;
    input.dest = 3;
    input.data = {};  // Пустые данные на остальных процессах
  }

  BarkalovaMStarMPI task(input);

  // Проверяем валидацию
  if (rank == 0) {
    ASSERT_TRUE(task.Validation());
  } else {
    ASSERT_TRUE(task.Validation());
  }

  ASSERT_TRUE(task.PreProcessing());
  bool run_result = task.Run();
  ASSERT_TRUE(run_result);
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 3) {
    // Только конечный получатель должен иметь данные
    EXPECT_EQ(output, std::vector<int>({5, 10, 15, 20, 25}));
  } else {
    // Центр и отправитель НЕ должны иметь данные в выходе
    EXPECT_TRUE(output.empty());
  }
}

// Тест 5: SEQ версия
TEST_F(BarkalovaMStarFuncTest, SeqVersion) {
  // SEQ тест должен работать только на 1 процессе
  int size = GetWorldSize();
  if (size != 1) {
    GTEST_SKIP() << "SEQ version requires exactly 1 process";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 0;
  input.data = {1, 3, 5, 7, 9};

  BarkalovaMStarSEQ task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();
  EXPECT_EQ(output, input.data);
}

// Тест 6: Некорректные параметры
TEST_F(BarkalovaMStarFuncTest, InvalidParameters) {
  int size = GetWorldSize();
  int rank = GetWorldRank();

  // Некорректный source
  StarMessage input1;
  if (rank == 0) {
    input1.source = -1;
    input1.dest = 0;
    input1.data = {1, 2, 3};
  } else {
    input1.source = -1;
    input1.dest = 0;
    input1.data = {};
  }

  BarkalovaMStarMPI task1(input1);
  EXPECT_FALSE(task1.Validation());

  // Некорректный dest
  StarMessage input2;
  if (rank == 0) {
    input2.source = 0;
    input2.dest = size + 10;
    input2.data = {1, 2, 3};
  } else {
    input2.source = 0;
    input2.dest = size + 10;
    input2.data = {};
  }

  BarkalovaMStarMPI task2(input2);
  EXPECT_FALSE(task2.Validation());
}

// Тест 7: Пустые данные
TEST_F(BarkalovaMStarFuncTest, EmptyData) {
  int size = GetWorldSize();
  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();  // Убираем предупреждение - используем переменную

  // Даже на процессе 0 данные пустые
  input.source = 0;
  input.dest = 1;
  input.data = {};  // Пустые данные на ВСЕХ процессах

  BarkalovaMStarMPI task(input);

  // Проверяем валидацию в зависимости от ранга
  bool validation_result = task.Validation();
  (void)rank;  // Используем переменную, чтобы убрать предупреждение

  ASSERT_TRUE(validation_result);
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();
  EXPECT_TRUE(output.empty());
}

// Тест 8: Большие данные
TEST_F(BarkalovaMStarFuncTest, LargeData) {
  int size = GetWorldSize();
  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  int rank = GetWorldRank();

  // Только процесс 0 инициализирует данные!
  if (rank == 0) {
    input.source = 0;
    input.dest = 1;
    input.data.resize(1000);

    for (size_t i = 0; i < input.data.size(); ++i) {
      input.data[i] = static_cast<int>(i * 2);
    }
  } else {
    input.source = 0;
    input.dest = 1;
    input.data = {};  // Пустые данные на остальных процессах
  }

  BarkalovaMStarMPI task(input);

  // Проверяем валидацию
  if (rank == 0) {
    ASSERT_TRUE(task.Validation());
  } else {
    ASSERT_TRUE(task.Validation());
  }

  ASSERT_TRUE(task.PreProcessing());
  bool run_result = task.Run();
  ASSERT_TRUE(run_result);
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 1) {
    // Только получатель должен иметь данные
    std::vector<int> expected(1000);
    for (size_t i = 0; i < expected.size(); ++i) {
      expected[i] = static_cast<int>(i * 2);
    }
    EXPECT_EQ(output, expected);
  } else {
    EXPECT_TRUE(output.empty());
  }
}

}  // namespace barkalova_m_star

/*
#include <gtest/gtest.h>
#include <mpi.h>

#include <array>
#include <cstddef>
#include <string>
#include <tuple>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"
#include "barkalova_m_star/seq/include/ops_seq.hpp"
#include "util/include/func_test_util.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_star {

using TestTypeSimple = std::string;

class BarkalovaMStarFuncTests : public ppc::util::BaseRunFuncTests<InType, OutType, TestTypeSimple> {
 public:
  static std::string PrintTestParam(const TestTypeSimple &test_param) {
    return test_param;
  }

 protected:
  void SetUp() override {
    int is_init = 0;
    MPI_Initialized(&is_init);

    if (is_init == 0) {
      input_message_ = {0, 0, 0, {}};
      expected_output_ = {};
      return;
    }

    int proc_count = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

    // Получаем имя задачи
    auto test_tuple = GetParam();
    std::string task_type = std::get<1>(test_tuple);  // Второй элемент - имя типа задачи

    // Если это SEQ задача и процессов больше 1 - пропускаем тест
    if (task_type.find("seq") != std::string::npos) {
      if (proc_count > 1) {
        skip_test_ = true;
        return;
      } else {
        skip_test_ = false;
      }
    } else {
      skip_test_ = false;
    }

    TestTypeSimple test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (test_name == "broadcast") {
      SetupBroadcastTest(proc_count);
    } else if (test_name == "center_to_peripheral") {
      SetupCenterToPeripheralTest(proc_count);
    } else if (test_name == "peripheral_to_center") {
      SetupPeripheralToCenterTest(proc_count);
    } else if (test_name == "peripheral_to_peripheral") {
      SetupPeripheralToPeripheralTest(proc_count);
    } else if (test_name == "seq_version") {
      SetupSEQVersionTest(proc_count);
    } else if (test_name == "empty_data") {
      SetupEmptyDataTest(proc_count);
    } else if (test_name == "large_data") {
      SetupLargeDataTest(proc_count);
    } else if (test_name == "different_center") {
      SetupDifferentCenterTest(proc_count);
    }
  }

  void SetupBroadcastTest(int proc_count) {
    (void)proc_count;
    input_message_ = {0, 0, 0, {1, 2, 3, 4, 5}};
    expected_output_ = {1, 2, 3, 4, 5};
  }

  void SetupCenterToPeripheralTest(int proc_count) {
    if (proc_count > 2) {
      input_message_ = {0, 0, 2, {10, 20, 30, 40}};
    } else if (proc_count > 1) {
      input_message_ = {0, 0, 1, {10, 20, 30, 40}};
    } else {
      input_message_ = {0, 0, 0, {10, 20, 30, 40}};
    }
    expected_output_ = {};
  }

  void SetupPeripheralToCenterTest(int proc_count) {
    if (proc_count > 1) {
      input_message_ = {0, 1, 0, {100, 200, 300}};
    } else {
      input_message_ = {0, 0, 0, {100, 200, 300}};
    }
    expected_output_ = {};
  }

  void SetupPeripheralToPeripheralTest(int proc_count) {
    if (proc_count > 3) {
      input_message_ = {0, 1, 3, {5, 10, 15, 20, 25}};
    } else if (proc_count > 2) {
      input_message_ = {0, 1, 2, {5, 10, 15, 20, 25}};
    } else if (proc_count > 1) {
      input_message_ = {0, 1, 1, {5, 10, 15, 20, 25}};
    } else {
      input_message_ = {0, 0, 0, {5, 10, 15, 20, 25}};
    }
    expected_output_ = {};
  }

  void SetupSEQVersionTest(int proc_count) {
    (void)proc_count;
    input_message_ = {0, 0, 0, {1, 3, 5, 7, 9}};
    expected_output_ = {1, 3, 5, 7, 9};
  }

  void SetupEmptyDataTest(int proc_count) {
    if (proc_count > 1) {
      input_message_ = {0, 0, 1, {}};
    } else {
      input_message_ = {0, 0, 0, {}};
    }
    expected_output_ = {};
  }

  void SetupLargeDataTest(int proc_count) {
    std::vector<int> large_data = CreateLargeData(1000);
    if (proc_count > 1) {
      input_message_ = {0, 0, 1, large_data};
    } else {
      input_message_ = {0, 0, 0, large_data};
    }
    expected_output_ = {};
  }

  void SetupDifferentCenterTest(int proc_count) {
    if (proc_count > 1) {
      input_message_ = {1, 1, 1, {2, 4, 6}};
    } else {
      input_message_ = {0, 0, 0, {2, 4, 6}};
    }
    expected_output_ = {2, 4, 6};
  }

  bool CheckTestOutputData(OutType &output_data) final {
  if (skip_test_) {
    // Для пропущенных тестов (SEQ в MPI) всегда возвращаем true
    return true;
  }

  int is_init = 0;
  MPI_Initialized(&is_init);

  if (is_init == 0) {
    return true;
  }

  int rank = 0, proc_count = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &proc_count);

  // Получаем имя задачи
  auto test_tuple = GetParam();
  std::string task_type = std::get<1>(test_tuple);

  // Если это SEQ задача и мы дошли сюда, значит proc_count = 1
  if (task_type.find("seq") != std::string::npos) {
    // SEQ версия всегда должна возвращать те же данные
    // Получаем входные данные из переменной input_message_
    return output_data == input_message_.data;
  }

  // Остальной код для MPI тестов остается без изменений
  TestTypeSimple test_name = std::get<static_cast<std::size_t>(ppc::util::GTestParamIndex::kTestParams)>(GetParam());

    if (proc_count == 1) {
      // SEQ версия
      return true;
    }

    // MPI версия
    if (test_name == "broadcast" || test_name == "seq_version" || test_name == "different_center") {
      return output_data == expected_output_;
    }

    if (test_name == "center_to_peripheral") {
      if (proc_count > 2) {
        if (rank == 2) return output_data == std::vector<int>{10, 20, 30, 40};
      } else if (proc_count > 1) {
        if (rank == 1) return output_data == std::vector<int>{10, 20, 30, 40};
      } else {
        if (rank == 0) return output_data == std::vector<int>{10, 20, 30, 40};
      }
      return output_data.empty();
    }

    if (test_name == "peripheral_to_center") {
      if (rank == 0) return output_data == std::vector<int>{100, 200, 300};
      return output_data.empty();
    }

    if (test_name == "peripheral_to_peripheral") {
      if (proc_count > 3) {
        if (rank == 3) return output_data == std::vector<int>{5, 10, 15, 20, 25};
      } else if (proc_count > 2) {
        if (rank == 2) return output_data == std::vector<int>{5, 10, 15, 20, 25};
      } else if (proc_count > 1) {
        if (rank == 1) return output_data == std::vector<int>{5, 10, 15, 20, 25};
      } else {
        if (rank == 0) return output_data == std::vector<int>{5, 10, 15, 20, 25};
      }
      return output_data.empty();
    }

    if (test_name == "empty_data") {
      return output_data.empty();
    }

    if (test_name == "large_data") {
      std::vector<int> large_data = CreateLargeData(1000);
      if (rank == 1) return output_data == large_data;
      return output_data.empty();
    }

    return output_data == expected_output_;
  }

  InType GetTestInputData() final {
    return input_message_;
  }

 private:
  std::vector<int> CreateLargeData(size_t size) {
    std::vector<int> result(size);
    for (size_t i = 0; i < size; ++i) {
      result[i] = static_cast<int>(i * 2);
    }
    return result;
  }

  InType input_message_;
  OutType expected_output_;
  bool skip_test_ = false;
};

namespace {

TEST_P(BarkalovaMStarFuncTests, StarTopologyTests) {
  ExecuteTest(GetParam());
}

const std::array<std::string, 8> kTestParam = {
    "broadcast",
    "center_to_peripheral",
    "peripheral_to_center",
    "peripheral_to_peripheral",
    "seq_version",
    "empty_data",
    "large_data",
    "different_center"
};

const auto kTestTasksList = std::tuple_cat(
    ppc::util::AddFuncTask<BarkalovaMStarMPI, InType>(kTestParam, PPC_SETTINGS_barkalova_m_star),
    ppc::util::AddFuncTask<BarkalovaMStarSEQ, InType>(kTestParam, PPC_SETTINGS_barkalova_m_star));

const auto kGtestValues = ppc::util::ExpandToValues(kTestTasksList);

const auto kPerfTestName = BarkalovaMStarFuncTests::PrintFuncTestName<BarkalovaMStarFuncTests>;

INSTANTIATE_TEST_SUITE_P(StarTopologyTests, BarkalovaMStarFuncTests, kGtestValues, kPerfTestName);

}  // namespace

}  // namespace barkalova_m_star
  */
