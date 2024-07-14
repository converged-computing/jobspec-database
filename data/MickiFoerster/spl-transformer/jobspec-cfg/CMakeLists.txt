cmake_minimum_required (VERSION 3.1)

project(
    SPL-Transformer
  VERSION 1.0
  LANGUAGES CXX)

find_program(BISON bison)
find_program(FLEX flex)

message(STATUS "flex: ${FLEX}")
message(STATUS "bison: ${BISON}")

# Exec Yacc
set(BISON_SRC           ${CMAKE_SOURCE_DIR}/parser.ypp)
set(BISON_OUT_CPP       parser.tab.cpp)
set(BISON_OUT_H         parser.tab.h)
set(BISON_OUT_DOT       parser.dot)
set(BISON_OUT_OUTPUT    parser.output)

add_custom_command(
	OUTPUT ${BISON_OUT_CPP} ${BISON_OUT_H} ${BISON_OUT_DOT} ${BISON_OUT_OUTPUT}
	COMMAND ${BISON} ARGS -v -g --warnings=error --defines=${BISON_OUT_H} ${BISON_SRC}
	DEPENDS ${BISON_SRC}
	)
set(GENERATED_SRCS    ${GENERATED_SRCS} 
	                  ${BISON_OUT_CPP} 
					  ${BISON_OUT_H} 
					  ${BISON_OUT_DOT} 
					  ${BISON_OUT_OUTPUT}
   )

# Exec Lex
set(FLEX_SRC ${CMAKE_SOURCE_DIR}/scanner.lpp)
set(FLEX_OUT_CPP scanner.cpp)
add_custom_command(
	OUTPUT ${FLEX_OUT_CPP}
	COMMAND ${FLEX} ARGS -o${FLEX_OUT_CPP} ${FLEX_SRC}
	DEPENDS ${FLEX_SRC}
	)
set(GENERATED_SRCS ${GENERATED_SRCS} ${FLEX_OUT_CPP})


# Tell CMake that some files are generated
set_source_files_properties(${GENERATED_SRCS} PROPERTIES GENERATED TRUE)

include_directories( ${CMAKE_SOURCE_DIR}  
           	         ${CMAKE_SOURCE_DIR}/prg-analysis/constant-propagation
           	         ${CMAKE_SOURCE_DIR}/prg-analysis/interval-analysis
           	         ${CMAKE_SOURCE_DIR}/prg-analysis/exclusive-read
				   )

set (EXESRC ${EXESRC} ${BISON_OUT_CPP})
set (EXESRC ${EXESRC} ${FLEX_OUT_CPP})
set (EXESRC ${EXESRC} symbol_table.cpp)
set (EXESRC ${EXESRC} ast_vertex.cpp)
set (EXESRC ${EXESRC} tau.cpp)
set (EXESRC ${EXESRC} sigma.cpp)
set (EXESRC ${EXESRC} rho.cpp)
set (EXESRC ${EXESRC} AST.cpp)
set (EXESRC ${EXESRC} unparser.cpp)
set (EXESRC ${EXESRC} prg-analysis/constant-propagation/constant-propagation.cpp)
set (EXESRC ${EXESRC} prg-analysis/interval-analysis/interval-analysis.cpp)
set (EXESRC ${EXESRC} prg-analysis/exclusive-read/exclusive-read.cpp)
set (EXESRC ${EXESRC} main.cpp)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O0 -g  ")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Werror -O0 -g  ")

add_executable (SPLc ${EXESRC})
 
