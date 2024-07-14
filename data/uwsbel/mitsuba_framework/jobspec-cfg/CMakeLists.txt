#=============================================================================
# CMake configuration file for the MitsubaGenerator library
#
#=============================================================================

cmake_minimum_required(VERSION 2.8)
project(Mitsuba_Generator)

# Location of additional CMAKE modules
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Policy CMP0046 warning when a dependency does not exist
# According to a dev from kitware The add_dependencies
# command is for inter-target ordering dependencies, not
# file-level dependencies.
# This warning appears in Cmake 3.0
IF(CMAKE_MAJOR_VERSION EQUAL 3)
    IF(COMMAND cmake_policy)
        cmake_policy(SET CMP0046 OLD)
    ENDIF(COMMAND cmake_policy)
ENDIF(CMAKE_MAJOR_VERSION EQUAL 3)
IF(APPLE)
    set(CMAKE_MACOSX_RPATH OFF)
ENDIF()

# ------------------------------------------------------------------------------
# Specify output locations for libraries and executables
# Set the build type depending on the platform
# ------------------------------------------------------------------------------
SET(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
SET(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

# ------------------------------------------------------------------------------
# Locate Dependencies
# ------------------------------------------------------------------------------

add_definitions(-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP)
add_definitions(-DTHRUST_HOST_SYSTEM=THRUST_HOST_SYSTEM_OMP)

find_package(ZLIB)
find_package(LibXml2)
find_package(Thrust)

#SET(PARTIO_NAMES ${PARTIO_NAMES} partio)  
#FIND_LIBRARY(PARTIO_LIBRARY NAMES ${PARTIO_NAMES} PATHS /usr/lib /usr/local/lib)

find_package(Chrono
			 COMPONENTS PARALLEL REQUIRED
             CONFIG 
)

IF(CMAKE_COMPILER_IS_GNUCC OR CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
	SET(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${CHRONO_CXX_FLAGS})
ENDIF()


# ------------------------------------------------------------------------------
# LIST THE FILES in the MitsubaGenerator LIBRARY
# ------------------------------------------------------------------------------


SET(MitsubaGenerator_SOURCES
    MitsubaGenerator.cpp
    sph_trianglemesh.cpp
    )

SET(MitsubaGenerator_HEADERS
    MitsubaGenerator.h
    sph_trianglemesh.h
    )


INCLUDE_DIRECTORIES(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/external
    ${CMAKE_BINARY_DIR}
    ${CHRONO_INCLUDE_DIRS}
    ${LIBXML2_INCLUDE_DIR}
	${ZLIB_INCLUDE_DIRS}
    ${THRUST_INCLUDE_DIR}
    )
SET(LIBRARIES  ${CHRONO_LIBRARIES}  ${ZLIB_LIBRARIES} )
# ------------------------------------------------------------------------------
# ADD THE MitsubaGenerator LIBRARY
# ------------------------------------------------------------------------------

ADD_LIBRARY(MitsubaGenerator SHARED
    ${MitsubaGenerator_HEADERS}
    ${MitsubaGenerator_SOURCES}
   )

SET_TARGET_PROPERTIES(MitsubaGenerator PROPERTIES
          COMPILE_FLAGS "${CH_BUILDFLAGS} -openmp"
        LINK_FLAGS "${CH_LINKERFLAG_EXE}")

 TARGET_LINK_LIBRARIES(MitsubaGenerator ${LIBXML2_LIBRARIES} ${LIBRARIES})

#TARGET_LINK_LIBRARIES(MitsubaGenerator Partio )

INSTALL(TARGETS MitsubaGenerator
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    )



ADD_SUBDIRECTORY(scenes)
