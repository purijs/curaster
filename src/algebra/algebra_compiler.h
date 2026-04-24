/**
 * @file algebra_compiler.h
 * @brief Shunting-yard compiler: expression string → Instruction array.
 *
 * Translates human-readable band-math expressions such as
 * "(B1 - B2) / (B1 + B2)" into a compact stack of Instruction objects
 * that the GPU algebra kernel evaluates independently for every output pixel.
 */
#pragma once

#include <string>
#include <vector>
#include "../../include/raster_core.h"  

/**
 * @brief Compile a raster algebra expression into a VM Instruction sequence.
 *
 * Supported syntax:
 *  - Band references:    B1, B2, … (1-indexed; mapped to 0-indexed internally)
 *  - Numeric literals:   42, 3.14, -1.5
 *  - Arithmetic:         +  -  *  /
 *  - Comparison:         >  <  >=  <=  ==  !=
 *  - Grouping:           ( )
 *
 * Band references are deduplicated: each unique band is added to @p band_map
 * once and all occurrences in the expression use its index in that map.
 *
 * @param expression  Algebra expression string to compile.
 * @param band_map    In/out: maps virtual slot → physical band index (0-based).
 *                    Pass an empty vector on first call; reuse across calls to
 *                    share the same band pool.
 * @return            Compiled Instruction sequence in postfix (RPN) order.
 */
std::vector<Instruction> compile_algebra_expression(const std::string& expression,
                                                     std::vector<int>&  band_map);
