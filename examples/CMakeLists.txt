find_package(Boost 1.60 COMPONENTS program_options REQUIRED)
INCLUDE_DIRECTORIES(${Boost_INCLUDE_DIR})

find_package(Ceres)

add_executable(mpi_daba_bal_dataset ./mpi_daba_bal_dataset.cu)
target_link_libraries(mpi_daba_bal_dataset DSfM ${Boost_LIBRARIES})

add_executable(mpi_dr_bal_dataset ./mpi_dr_bal_dataset.cu)
target_link_libraries(mpi_dr_bal_dataset DSfM ExampleDouglasRachford ${Boost_LIBRARIES})

add_executable(mpi_admm_bal_dataset ./mpi_admm_bal_dataset.cu)
target_link_libraries(mpi_admm_bal_dataset DSfM ExampleADMM ${Boost_LIBRARIES})

add_executable(mem_comm_bal_dataset ./mem_comm_bal_dataset.cu)
target_link_libraries(mem_comm_bal_dataset DSfM ExampleADMM ExampleDouglasRachford)

if (CERES_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_executable(ceres_bal_dataset ./ceres_bal_dataset.cpp)
    target_link_libraries(ceres_bal_dataset DSfM ${CERES_LIBRARIES} ${Boost_LIBRARIES})
endif()

add_subdirectory(./admm)
add_subdirectory(./douglas_rachford)