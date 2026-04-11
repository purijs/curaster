/**
 * @file algebra_compiler.cpp
 * @brief Shunting-yard compiler that converts algebra expression strings to
 *        postfix Instruction sequences for the GPU stack-based VM.
 */
#include "algebra_compiler.h"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Operator precedence ──────────────────────────────────────────────────────

/// Return the precedence level of an infix operator token (0 = not an operator).
static int operator_precedence(const std::string& token) {
    if (token == "+" || token == "-") { return 1; }
    if (token == "*" || token == "/") { return 2; }
    return 0;
}

// ─── Opcode emission ──────────────────────────────────────────────────────────

/// Map an operator token string to its Opcode and append the instruction.
static void emit_operator(const std::string& token, std::vector<Instruction>& output) {
    Opcode op;
    if      (token == "+")  { op = OP_ADD; }
    else if (token == "-")  { op = OP_SUB; }
    else if (token == "*")  { op = OP_MUL; }
    else if (token == "/")  { op = OP_DIV; }
    else if (token == ">")  { op = OP_GT;  }
    else if (token == "<")  { op = OP_LT;  }
    else if (token == ">=") { op = OP_GTE; }
    else if (token == "<=") { op = OP_LTE; }
    else if (token == "==") { op = OP_EQ;  }
    else if (token == "!=") { op = OP_NEQ; }
    else { return; } // Unrecognised token — skip silently.

    output.push_back({op, 0.0f, -1});
}

// ─── Tokeniser ────────────────────────────────────────────────────────────────

/**
 * @brief Split the expression into atomic tokens.
 *
 * Single-character operators, brackets, and commas are emitted as standalone
 * tokens; all other characters accumulate into word tokens (band refs, numbers).
 */
static std::vector<std::string> tokenise(const std::string& expression) {
    std::vector<std::string> tokens;
    std::string current_word;

    auto flush_word = [&]() {
        if (!current_word.empty()) {
            tokens.push_back(current_word);
            current_word.clear();
        }
    };

    for (char ch : expression) {
        if (std::isspace(static_cast<unsigned char>(ch))) {
            flush_word();
            continue;
        }

        // These characters are always single-token delimiters.
        bool is_single_char_token =
            ch == '+' || ch == '-' || ch == '*' || ch == '/'
         || ch == '(' || ch == ')' || ch == ','
         || ch == '<' || ch == '>' || ch == '=' || ch == '!';

        if (is_single_char_token) {
            flush_word();
            tokens.push_back(std::string(1, ch));
        } else {
            current_word += ch;
        }
    }
    flush_word();

    // Merge two-character comparison tokens (e.g. ">" followed by "=").
    std::vector<std::string> merged;
    for (size_t i = 0; i < tokens.size(); ) {
        if (i + 1 < tokens.size()) {
            std::string pair = tokens[i] + tokens[i + 1];
            if (pair == ">=" || pair == "<=" || pair == "==" || pair == "!=") {
                merged.push_back(pair);
                i += 2;
                continue;
            }
        }
        merged.push_back(tokens[i++]);
    }
    return merged;
}

// ─── compile_algebra_expression ──────────────────────────────────────────────

std::vector<Instruction> compile_algebra_expression(const std::string& expression,
                                                     std::vector<int>&  band_map) {
    std::vector<std::string> tokens = tokenise(expression);

    // Shunting-yard algorithm — builds postfix output from infix tokens.
    std::vector<Instruction> output_queue;
    std::vector<std::string> operator_stack;

    for (const std::string& token : tokens) {

        // ── Band reference: B1, B2, … ────────────────────────────────────
        if (token.size() >= 2 && token[0] == 'B' && std::isdigit(token[1])) {
            int physical_index = std::stoi(token.substr(1)) - 1; // Convert to 0-based

            // Deduplicate: reuse existing slot or append a new one.
            auto existing = std::find(band_map.begin(), band_map.end(), physical_index);
            int  slot_index;
            if (existing != band_map.end()) {
                slot_index = static_cast<int>(std::distance(band_map.begin(), existing));
            } else {
                slot_index = static_cast<int>(band_map.size());
                band_map.push_back(physical_index);
            }

            output_queue.push_back({OP_LOAD_BAND, 0.0f, slot_index});
            continue;
        }

        // ── Numeric literal ───────────────────────────────────────────────
        bool is_number = std::isdigit(static_cast<unsigned char>(token[0]))
                      || (token[0] == '-' && token.size() > 1)
                      || token[0] == '.';
        if (is_number) {
            float value = std::stof(token);
            output_queue.push_back({OP_LOAD_CONST, value, -1});
            continue;
        }

        // ── Left parenthesis ──────────────────────────────────────────────
        if (token == "(") {
            operator_stack.push_back(token);
            continue;
        }

        // ── Right parenthesis: flush operators to matching left paren ─────
        if (token == ")") {
            while (!operator_stack.empty() && operator_stack.back() != "(") {
                emit_operator(operator_stack.back(), output_queue);
                operator_stack.pop_back();
            }
            if (!operator_stack.empty()) {
                operator_stack.pop_back(); // Discard the "("
            }
            continue;
        }

        // ── Infix operator ────────────────────────────────────────────────
        if (operator_precedence(token) > 0) {
            while (!operator_stack.empty()
                   && operator_precedence(operator_stack.back()) >= operator_precedence(token)) {
                emit_operator(operator_stack.back(), output_queue);
                operator_stack.pop_back();
            }
            operator_stack.push_back(token);
        }
    }

    // Drain the operator stack.
    while (!operator_stack.empty()) {
        if (operator_stack.back() != "(") {
            emit_operator(operator_stack.back(), output_queue);
        }
        operator_stack.pop_back();
    }

    return output_queue;
}
