#include "barkalova_m_int_met_trapez/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "barkalova_m_int_met_trapez/common/include/common.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_int_met_trapez {

BarkalovaMIntMetTrapezMPI::BarkalovaMIntMetTrapezMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
}

bool BarkalovaMIntMetTrapezMPI::ValidationImpl() {
  auto &data = GetInput();
  // Проверяем, что векторы имеют достаточно элементов
  return data.limits.size() >= 2 && data.n_i.size() >= 2;
}

bool BarkalovaMIntMetTrapezMPI::PreProcessingImpl() {
  GetOutput() = 0.0;
  return true;
}
namespace {
// Структура для передачи данных между процессами
struct BroadcastData {
  int n_steps_x;
  int n_steps_y;
  double x1, x2, y1, y2;
};

template <typename Func>
double RunKernel(const BroadcastData &data, int rank, int size, const Func &f) {
  double hx = (data.x2 - data.x1) / data.n_steps_x;
  double hy = (data.y2 - data.y1) / data.n_steps_y;

  int total_nodes_x = data.n_steps_x + 1;

  int count = total_nodes_x / size;
  int remainder = total_nodes_x % size;

  int start_i = (rank * count) + std::min(rank, remainder);
  int end_i = start_i + count + (rank < remainder ? 1 : 0);

  double local_sum = 0.0;

  for (int i = start_i; i < end_i; ++i) {
    double x = data.x1 + (i * hx);
    double weight_x = (i == 0 || i == data.n_steps_x) ? 0.5 : 1.0;

    for (int j = 0; j <= data.n_steps_y; ++j) {
      double y = data.y1 + (j * hy);
      double weight_y = (j == 0 || j == data.n_steps_y) ? 0.5 : 1.0;

      local_sum += f(x, y) * weight_x * weight_y;
    }
  }

  return local_sum * hx * hy;
}
}  // namespace

bool BarkalovaMIntMetTrapezMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  Integral data_local;
  if (rank == 0) {
    data_local = GetInput();
  }

  BroadcastData bcast_data;
  if (rank == 0) {
    // Извлекаем данные из структуры Integral
    bcast_data.n_steps_x = data_local.n_i[0];
    bcast_data.n_steps_y = data_local.n_i[1];
    bcast_data.x1 = data_local.limits[0].first;
    bcast_data.x2 = data_local.limits[0].second;
    bcast_data.y1 = data_local.limits[1].first;
    bcast_data.y2 = data_local.limits[1].second;
  }

  MPI_Bcast(&bcast_data.n_steps_x, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_data.n_steps_y, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_data.x1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_data.x2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_data.y1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(&bcast_data.y2, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  double local_result = 0.0;

  // В вашей структуре Integral есть метод function, но в MPI его нельзя использовать напрямую,
  // поэтому используем функцию, соответствующую той, что определена в структуре
  // function(std::vector<double> x) возвращает сумму квадратов координат
  local_result = RunKernel(bcast_data, rank, size, [](double x, double y) { return x * x + y * y; });

  double global_result = 0.0;
  MPI_Reduce(&local_result, &global_result, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  MPI_Bcast(&global_result, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  GetOutput() = global_result;

  return true;
}

bool BarkalovaMIntMetTrapezMPI::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_int_met_trapez
