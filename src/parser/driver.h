#pragma once

#include <map>
#include <string>

#include "generated/parser.h"

// Give Flex the prototype of yylex we want ...
#define YY_DECL yy::parser::symbol_type yylex(driver& drv)
// ... and declare it for the parser's sake.
YY_DECL;

// Conducting the whole scanning and parsing of cfg and bnd files.
class driver
{
public:
	enum class start_type
	{
		cfg,
		bnd,
		upp,
		none
	};

	// Whether to generate parser debug traces.
	bool trace_parsing;
	// Whether to generate scanner debug traces.
	bool trace_scanning;
	// The name of the file being parsed.
	std::string file;
	// Denotes if we parse cfg or bnd file
	start_type start;

	// Handling the scanner.
	void scan_begin();
	void scan_end();
	// The token's location used by the scanner.
	yy::location location;

	driver();

	std::map<std::string, float> variables;
	std::map<std::string, float> constants;
	std::vector<node_t> nodes;
	std::vector<ext_inp_t> external_inputs;

	void register_variable(std::string name, expr_ptr expr);
	void register_constant(std::string name, expr_ptr expr);
	void register_node(std::string name, node_attr_list_t expr);
	void register_node_attribute(std::string node, std::string name, expr_ptr expr);
	void register_node_istate(std::string node, expr_ptr expr_l, expr_ptr expr_r, int value_l);


	void register_external_input(std::string name);
	void register_external_input(std::string name, expr_ptr expr);
	void register_external_input_expression_attribute(std::string name, expr_ptr);
	void register_external_input_expression_attribute(ext_inp_t external_inp, expr_ptr expr);


	// int parse(std::string bnd_file, std::string cfg_file);
	int parse(std::string bnd_file, std::string cfg_file,std::string  upp_file);

	// Run the parser on the file.
	// Return 0 on success.
	int parse_one(std::string file);
};
