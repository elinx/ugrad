pybind11_add_module(pyugrad pyugrad.cpp)
target_link_libraries(pyugrad PRIVATE ugrad)
