###############################################################################
# (c) Copyright 2020 CERN for the benefit of the LHCb Collaboration           #
###############################################################################
include_directories(include)
# include_directories(${CMAKE_SOURCE_DIR}/main/include)
# include_directories(${CMAKE_SOURCE_DIR}/external)
# include_directories(${CMAKE_SOURCE_DIR}/stream/gear/include)

file(GLOB backend_sources "src/*cpp")

allen_add_host_library(Backend STATIC ${backend_sources})
