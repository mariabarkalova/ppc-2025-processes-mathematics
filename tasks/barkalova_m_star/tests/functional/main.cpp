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
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized == 0) {
      MPI_Init(nullptr, nullptr);
    }
  }

  static void TearDownTestSuite() {
    int is_mpi_initialized = 0;
    MPI_Initialized(&is_mpi_initialized);
    if (is_mpi_initialized != 0) {
      int is_mpi_finalized = 0;
      MPI_Finalized(&is_mpi_finalized);
      if (is_mpi_finalized == 0) {
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

// от центра к центру
TEST_F(BarkalovaMStarFuncTest, CenterToCenter) {
  int rank = GetWorldRank();
  // int size = GetWorldSize();

  StarMessage input;
  input.source = 0;
  input.dest = 0;

  std::vector<int> test_data = {42, 15, 73, 29, 88};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 0) {
    EXPECT_EQ(output, test_data);
  } else {
    EXPECT_TRUE(output.empty());
  }
}

// от центра к периферийному узлу
TEST_F(BarkalovaMStarFuncTest, CenterToPeripheral) {
  int rank = GetWorldRank();
  int size = GetWorldSize();

  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 1;

  std::vector<int> test_data = {100, 200, 300, 400, 500};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 1) {
    EXPECT_EQ(output, test_data);
  } else {
    EXPECT_TRUE(output.empty());
  }
}

// от периферийного узла центру
TEST_F(BarkalovaMStarFuncTest, PeripheralToCenter) {
  int rank = GetWorldRank();
  int size = GetWorldSize();

  if (size < 2) {
    GTEST_SKIP() << "Need at least 2 processes";
  }

  StarMessage input;
  input.source = 1;
  input.dest = 0;

  std::vector<int> test_data = {777, 888, 999};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 0) {
    EXPECT_EQ(output, test_data);
  } else {
    EXPECT_TRUE(output.empty());
  }
}

// между двумя периферийными узлами через центр
TEST_F(BarkalovaMStarFuncTest, PeripheralToPeripheralThroughCenter) {
  int rank = GetWorldRank();
  int size = GetWorldSize();

  if (size < 3) {
    GTEST_SKIP() << "Need at least 3 processes";
  }

  StarMessage input;
  input.source = 1;
  input.dest = 2;

  std::vector<int> test_data = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  input.data = test_data;

  BarkalovaMStarMPI task(input);

  ASSERT_TRUE(task.Validation());
  ASSERT_TRUE(task.PreProcessing());
  ASSERT_TRUE(task.Run());
  ASSERT_TRUE(task.PostProcessing());

  auto output = task.GetOutput();

  if (rank == 2) {
    EXPECT_EQ(output, test_data);
  } else {
    EXPECT_TRUE(output.empty());
  }
}

TEST_F(BarkalovaMStarFuncTest, ValidationTests) {
  int size = GetWorldSize();

  // Корректные данные
  StarMessage valid_input;
  valid_input.source = 0;
  valid_input.dest = (size > 1) ? 1 : 0;
  valid_input.data = {1, 2, 3};

  BarkalovaMStarMPI valid_task(valid_input);
  EXPECT_TRUE(valid_task.Validation());

  StarMessage invalid_source;
  invalid_source.source = -1;
  invalid_source.dest = 0;
  invalid_source.data = {1, 2, 3};

  BarkalovaMStarMPI invalid_source_task(invalid_source);
  EXPECT_FALSE(invalid_source_task.Validation());

  StarMessage invalid_dest;
  invalid_dest.source = 0;
  invalid_dest.dest = size + 100;
  invalid_dest.data = {1, 2, 3};

  BarkalovaMStarMPI invalid_dest_task(invalid_dest);
  EXPECT_FALSE(invalid_dest_task.Validation());
}

// SEQ версия
TEST_F(BarkalovaMStarFuncTest, SequentialVersion) {
  int size = GetWorldSize();
  if (size != 1) {
    GTEST_SKIP() << "SEQ version requires exactly 1 process";
  }

  StarMessage input;
  input.source = 0;
  input.dest = 0;

  std::vector<int> test_data;
  test_data.reserve(100);
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
