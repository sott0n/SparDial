# Configure lit configuration files
configure_lit_site_cfg(
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
    MAIN_CONFIG
    ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

# Set test dependencies
set(SPARDIAL_TEST_DEPENDS
    SparDialPythonModules
)

# Add spardial-opt if it exists
if(TARGET spardial-opt)
    list(APPEND SPARDIAL_TEST_DEPENDS spardial-opt)
endif()

# Add FileCheck if available (from LLVM tools)
if(TARGET FileCheck)
    list(APPEND SPARDIAL_TEST_DEPENDS FileCheck)
endif()

# Create the check-spardial target
add_lit_testsuite(check-spardial "Running the SparDial regression tests"
    ${CMAKE_CURRENT_BINARY_DIR}
    DEPENDS ${SPARDIAL_TEST_DEPENDS}
)
set_target_properties(check-spardial PROPERTIES FOLDER "Tests")

# Add test suites
add_lit_testsuites(SPARDIAL ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${SPARDIAL_TEST_DEPENDS}
)
