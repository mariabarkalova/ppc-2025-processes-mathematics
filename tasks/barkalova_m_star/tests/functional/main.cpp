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

//  Отправка данных от центра (0) центру (0) - широковещание всем
TEST_F(BarkalovaMStarFuncTest, CenterToCenterBroadcast) {
  int size = GetWorldSize();
  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 0;

  // Генерируем случайные данные
  std::vector<int> test_data = {42, 15, 73, 29, 88};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // После HandleSameSourceDestination с broadcast все процессы должны иметь данные
  auto output = task.GetOutput();
  EXPECT_EQ(output, test_data);
}

// Отправка от центра к периферийному узлу (и broadcast всем)
TEST_F(BarkalovaMStarFuncTest, CenterToPeripheralWithBroadcast) {
  int size = GetWorldSize();
  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 1;  // Отправляем процессу 1

  std::vector<int> test_data = {100, 200, 300, 400, 500};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // После HandleDifferentSourceDestination с MPI_Bcast ВСЕ процессы должны получить данные
  auto output = task.GetOutput();
  EXPECT_EQ(output, test_data);
}

// Отправка от периферийного узла центру
TEST_F(BarkalovaMStarFuncTest, PeripheralToCenter) {
  int size = GetWorldSize();
  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  input.source = 2;  // Процесс 2 отправляет
  input.dest = 0;    // Процессу 0 (центру)

  std::vector<int> test_data = {777, 888, 999};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // После HandleDifferentSourceDestination с MPI_Bcast ВСЕ процессы должны получить данные
  auto output = task.GetOutput();
  EXPECT_EQ(output, test_data);
}

// Отправка между двумя периферийными узлами через центр
TEST_F(BarkalovaMStarFuncTest, PeripheralToPeripheralThroughCenter) {
  int size = GetWorldSize();
  if (size < 4) {
    GTEST_SKIP() << "Need at least 4 processes";
  }

  StarMessage input;
  input.source = 1;  // От процессора 1
  input.dest = 3;    // К процессору 3

  std::vector<int> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // После HandleDifferentSourceDestination с MPI_Bcast ВСЕ процессы должны получить данные
  auto output = task.GetOutput();
  EXPECT_EQ(output, test_data);
}

// Отправка самому себе (source == dest)
TEST_F(BarkalovaMStarFuncTest, SendToSelf) {
  int size = GetWorldSize();
  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  // Каждый процесс отправляет сам себе
  int rank = GetWorldRank();
  input.source = rank;
  input.dest = rank;

  std::vector<int> self_data = {rank * 10 + 1, rank * 10 + 2, rank * 10 + 3};
  input.data = self_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // В этом случае каждый процесс должен получить СВОИ данные
  auto output = task.GetOutput();
  EXPECT_EQ(output, self_data);
}

/*// Тест 7: Малое количество процессов (< 3)
TEST_F(BarkalovaMStarFuncTest, FewProcesses) {
  int size = GetWorldSize();

  StarMessage input;
  input.source = 0;
  input.dest = size > 1 ? 1 : 0;
  input.data = {1, 2, 3};

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  // При < 3 процессах должен вернуться {INT_MAX}
  if (size < 3) {
    auto output = task.GetOutput();
    EXPECT_EQ(output.size(), 1);
    EXPECT_EQ(output[0], INT_MAX);
  } else {
    // При >= 3 процессах нормальная работа
    auto output = task.GetOutput();
    EXPECT_EQ(output, std::vector<int>({1, 2, 3}));
  }
}*/

// Тест 8: Проверка корректности валидации
TEST_F(BarkalovaMStarFuncTest, ValidationTests) {
  int size = GetWorldSize();

  // Корректные данные
  StarMessage valid_input;
  valid_input.source = 0;
  valid_input.dest = (size > 1) ? 1 : 0;
  valid_input.data = {1, 2, 3};

  BarkalovaMStarMPI valid_task(valid_input);
  EXPECT_TRUE(valid_task.Validation());

  // Некорректный source (отрицательный)
  StarMessage invalid_source;
  invalid_source.source = -1;
  invalid_source.dest = 0;
  invalid_source.data = {1, 2, 3};

  BarkalovaMStarMPI invalid_source_task(invalid_source);
  EXPECT_FALSE(invalid_source_task.Validation());

  // Некорректный dest (больше количества процессов)
  StarMessage invalid_dest;
  invalid_dest.source = 0;
  invalid_dest.dest = size + 100;
  invalid_dest.data = {1, 2, 3};

  BarkalovaMStarMPI invalid_dest_task(invalid_dest);
  EXPECT_FALSE(invalid_dest_task.Validation());

  // Некорректный source и dest (оба за пределами)
  StarMessage invalid_both;
  invalid_both.source = size + 50;
  invalid_both.dest = size + 100;
  invalid_both.data = {1, 2, 3};

  BarkalovaMStarMPI invalid_both_task(invalid_both);
  EXPECT_FALSE(invalid_both_task.Validation());
}

// Тест 9: SEQ версия (для сравнения)
TEST_F(BarkalovaMStarFuncTest, SequentialVersion) {
  // SEQ тест должен работать только на 1 процессе
  int size = GetWorldSize();
  if (size != 1) {
    GTEST_SKIP() << "SEQ version requires exactly 1 process";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 0;

  std::vector<int> test_data;
  for (int i = 0; i < 100; ++i) {
    test_data.push_back(i * i);
  }
  input.data = test_data;

  BarkalovaMStarSEQ seq_task(input);

  ASSERT_TRUE(seq_task.Validation());
  ASSERT_TRUE(seq_task.PreProcessing());
  ASSERT_TRUE(seq_task.Run());
  ASSERT_TRUE(seq_task.PostProcessing());

  auto output = seq_task.GetOutput();
  EXPECT_EQ(output, test_data);
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
