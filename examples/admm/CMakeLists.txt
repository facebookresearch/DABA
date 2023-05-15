add_library(ExampleADMM ./admm_problem.h ./admm_problem.cu)
target_link_libraries(ExampleADMM PUBLIC DSfM)