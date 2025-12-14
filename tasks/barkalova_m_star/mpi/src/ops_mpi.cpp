/*
#include "barkalova_m_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <numeric>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"
#include "util/include/util.hpp"

namespace barkalova_m_star {

BarkalovaMStarMPI::BarkalovaMStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = 0;
}

bool BarkalovaMStarMPI::ValidationImpl() {
  return (GetInput() > 0) && (GetOutput() == 0);
}

bool BarkalovaMStarMPI::PreProcessingImpl() {
  GetOutput() = 2 * GetInput();
  return GetOutput() > 0;
}

bool BarkalovaMStarMPI::RunImpl() {
  auto input = GetInput();
  if (input == 0) {
    return false;
  }

  for (InType i = 0; i < GetInput(); i++) {
    for (InType j = 0; j < GetInput(); j++) {
      for (InType k = 0; k < GetInput(); k++) {
        std::vector<InType> tmp(i + j + k, 1);
        GetOutput() += std::accumulate(tmp.begin(), tmp.end(), 0);
        GetOutput() -= i + j + k;
      }
    }
  }

  const int num_threads = ppc::util::GetNumThreads();
  GetOutput() *= num_threads;

  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (rank == 0) {
    GetOutput() /= num_threads;
  } else {
    int counter = 0;
    for (int i = 0; i < num_threads; i++) {
      counter++;
    }

    if (counter != 0) {
      GetOutput() /= counter;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  return GetOutput() > 0;
}

bool BarkalovaMStarMPI::PostProcessingImpl() {
  GetOutput() -= GetInput();
  return GetOutput() > 0;
}

}  // namespace barkalova_m_star
*/

#include "barkalova_m_star/mpi/include/ops_mpi.hpp"

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <vector>

#include "barkalova_m_star/common/include/common.hpp"

namespace barkalova_m_star {

BarkalovaMStarMPI::BarkalovaMStarMPI(const InType &in) {
  SetTypeOfTask(GetStaticTypeOfTask());
  GetInput() = in;
  GetOutput() = {};
}

bool BarkalovaMStarMPI::ValidationImpl() {
  int size = 0;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  const auto &input = GetInput();

  // Центр всегда 0, проверяем только source и dest
  if (input.source < 0 || input.source >= size) {
    return false;
  }

  if (input.dest < 0 || input.dest >= size) {
    return false;
  }

  // Если мы не процесс 0, данные должны быть пустыми
  int rank = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  if (rank != 0 && !input.data.empty()) {
    return false;  // Только процесс 0 может иметь данные
  }

  return true;
}

bool BarkalovaMStarMPI::PreProcessingImpl() {
  GetOutput() = {};
  return true;
}

bool BarkalovaMStarMPI::RunImpl() {
  int rank = 0;
  int size = 0;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Центр всегда 0
  const int CENTER = 0;

  // ШАГ 1: Получаем параметры от процесса 0
  int source, dest;
  uint64_t data_size = 0;
  std::vector<int> data;

  if (rank == 0) {
    // Только процесс 0 знает исходные данные
    const auto &input = GetInput();
    source = input.source;
    dest = input.dest;
    data = input.data;
    data_size = static_cast<uint64_t>(data.size());
  }

  // Рассылаем параметры source и dest всем процессам
  MPI_Bcast(&source, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&dest, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(&data_size, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

  // Подготавливаем буфер для данных
  if (rank != 0) {
    data.resize(data_size);
  }

  GetOutput().clear();

  //  1: Широковещание (от центра всем)
  if (source == CENTER && dest == CENTER) {
    // Процесс 0 рассылает данные всем
    MPI_Bcast(data.data(), static_cast<int>(data_size), MPI_INT, 0, MPI_COMM_WORLD);
    GetOutput() = data;  // Все процессы получают данные
  }
  // 2: Отправка от центра к периферии
  else if (source == CENTER && dest != CENTER) {
    // Процесс 0 отправляет данные получателю
    if (rank == 0) {
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, dest, 0, MPI_COMM_WORLD);
    } else if (rank == dest) {
      GetOutput().resize(data_size);
      MPI_Recv(GetOutput().data(), static_cast<int>(data_size), MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  //  3: Отправка от периферии к центру
  else if (source != CENTER && dest == CENTER) {
    // Отправитель получает данные от процесса 0 (если он не 0)
    if (rank == 0 && source != 0) {
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, source, 1, MPI_COMM_WORLD);
    } else if (rank == source && source != 0) {
      MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Отправитель отправляет данные центру (процессу 0)
    if (rank == source) {
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, 0, 2, MPI_COMM_WORLD);
    } else if (rank == 0) {
      GetOutput().resize(data_size);
      MPI_Recv(GetOutput().data(), static_cast<int>(data_size), MPI_INT, source, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }
  // СЦЕНАРИЙ 4: Отправка между периферийными узлами через центр
  else if (source != CENTER && dest != CENTER) {
    // Процесс 0 отправляет данные отправителю
    if (rank == 0 && source != 0) {
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, source, 3, MPI_COMM_WORLD);
    } else if (rank == source && source != 0) {
      MPI_Recv(data.data(), static_cast<int>(data_size), MPI_INT, 0, 3, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }

    // Отправитель отправляет данные центру (процессу 0)
    if (rank == source) {
      MPI_Send(data.data(), static_cast<int>(data_size), MPI_INT, 0, 4, MPI_COMM_WORLD);
    } else if (rank == 0) {
      std::vector<int> temp_buffer(data_size);
      MPI_Recv(temp_buffer.data(), static_cast<int>(data_size), MPI_INT, source, 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

      // Центр отправляет данные получателю
      MPI_Send(temp_buffer.data(), static_cast<int>(data_size), MPI_INT, dest, 5, MPI_COMM_WORLD);
    } else if (rank == dest) {
      GetOutput().resize(data_size);
      MPI_Recv(GetOutput().data(), static_cast<int>(data_size), MPI_INT, 0, 5, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
  }

  // Синхронизация
  MPI_Barrier(MPI_COMM_WORLD);
  return true;
}

bool BarkalovaMStarMPI::PostProcessingImpl() {
  return true;
}

}  // namespace barkalova_m_star
