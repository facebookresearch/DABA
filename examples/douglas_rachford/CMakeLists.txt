add_library(ExampleDouglasRachford ./douglas_rachford_problem.h ./douglas_rachford_problem.cu)
target_link_libraries(ExampleDouglasRachford PUBLIC DSfM)