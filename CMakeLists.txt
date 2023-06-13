##
#  CMake script for the Eulero program:
##

# Set the name of the project and target:
SET(TARGET "main")

# Declare all source files the target consists of. Here, this is only
# the one step-X.cc file, but as you expand your project you may wish
# to add other source files as well. If your project becomes much larger,
# you may want to either replace the following statement by something like
  #  FILE(GLOB_RECURSE TARGET_SRC  "source/*.cc")
   # FILE(GLOB_RECURSE TARGET_INC  "include/*.h")

# or switch altogether to the large project CMakeLists.txt file discussed
# in the "CMake in user projects" page accessible from the "User info"
# page of the documentation.
SET(TARGET_SRC
      ${TARGET}.cpp

 #   main.cpp
    parameters.cpp
    parameters.h
    inputfile.prm
    inputfile_bstep.prm
    inputfile_cylinder.prm
    inputfile_DMR.prm
    inputfile_fstep.prm
    inputfile_sod.prm
    inputfile_2Driemann.prm
    equationdata.cpp
    equationdata.h
    euleroperator.cpp
    euleroperator.h
    operations.h
    eulerproblem.cpp
   eulerproblem.h
  )


# Usually, you will not need to modify anything beyond this point...

CMAKE_MINIMUM_REQUIRED(VERSION 3.1.0)

FIND_PACKAGE(deal.II 9.3.0
  HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
  )
IF(NOT ${deal.II_FOUND})
  MESSAGE(FATAL_ERROR "\n"
    "*** Could not locate a (sufficiently recent) version of deal.II. ***\n\n"
    "You may want to either pass a flag -DDEAL_II_DIR=/path/to/deal.II to cmake\n"
    "or set an environment variable \"DEAL_II_DIR\" that contains this path."
    )
ENDIF()

#
# Are all dependencies fulfilled?
#
IF(NOT DEAL_II_WITH_P4EST)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_P4EST = ON
This feature is OFF in your installation but is required for this program."
          )
ENDIF()


IF(NOT DEAL_II_WITH_MPI)
  MESSAGE(FATAL_ERROR "
Error! The deal.II library found at ${DEAL_II_PATH} was not configured with
    DEAL_II_WITH_MPI = ON
This feature is OFF in your installation but is required for this program."
          )
ENDIF()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
PROJECT(Project)
DEAL_II_INVOKE_AUTOPILOT()
