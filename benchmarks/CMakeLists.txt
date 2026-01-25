# SparDial Benchmark Targets

# Find Python
find_package(Python3 COMPONENTS Interpreter REQUIRED)

# Python packages are installed to python_packages/spardial
# The 'spardial' module is inside that directory
set(SPARDIAL_PYTHON_PKG_DIR "${SPARDIAL_BINARY_DIR}/python_packages/spardial")

# Default benchmark run (comprehensive)
add_custom_target(spardial-benchmark
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks
            --benchmarks spmv,spmm,add,numpy_spmv
            --sizes 100x100,500x500,1000x1000,2000x2000,5000x5000
            --sparsities 0.9,0.95,0.99
            --formats csr
            --progress
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running SparDial benchmarks"
)

# Quick benchmark for sanity check
add_custom_target(spardial-benchmark-quick
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks --quick --progress
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running quick SparDial benchmarks"
)

# Medium benchmark (faster than full, more comprehensive than quick)
add_custom_target(spardial-benchmark-medium
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks
            --benchmarks spmv,spmm,add,numpy_spmv
            --sizes 500x500,1000x1000,2000x2000
            --sparsities 0.9,0.95,0.99
            --formats csr
            --progress
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running SparDial benchmarks (medium)"
)

# JSON output for CI
add_custom_target(spardial-benchmark-json
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks
            --benchmarks spmv,spmm,add,numpy_spmv
            --sizes 100x100,500x500,1000x1000,2000x2000,5000x5000
            --sparsities 0.9,0.95,0.99
            --formats csr
            --output json
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running SparDial benchmarks (JSON output)"
)

# CSV output for analysis
add_custom_target(spardial-benchmark-csv
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks
            --benchmarks spmv,spmm,add,numpy_spmv
            --sizes 100x100,500x500,1000x1000,2000x2000,5000x5000
            --sparsities 0.9,0.95,0.99
            --formats csr
            --output csv
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running SparDial benchmarks (CSV output)"
)

# NumPy backend SpMV benchmark only
add_custom_target(spardial-benchmark-numpy
    COMMAND ${CMAKE_COMMAND} -E env
        PYTHONPATH=${SPARDIAL_PYTHON_PKG_DIR}:${SPARDIAL_SOURCE_DIR}:$ENV{PYTHONPATH}
        ${Python3_EXECUTABLE} -m benchmarks
            --benchmarks numpy_spmv
            --sizes 100x100,500x500,1000x1000
            --sparsities 0.9,0.95,0.99
            --formats csr
            --progress
    WORKING_DIRECTORY ${SPARDIAL_SOURCE_DIR}
    DEPENDS SparDialPythonModules
    COMMENT "Running NumPy backend SpMV benchmarks"
)
