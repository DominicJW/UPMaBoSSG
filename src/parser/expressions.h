#pragma once

#include <memory>
#include <string>
#include <vector>
#include <set>
#include "../state_word.h"
#include "set_hash.h"
#include <unordered_map>


class expression;
class driver;

using expr_ptr = std::unique_ptr<expression>;

enum class operation
{
	PLUS,
	MINUS,
	STAR,
	SLASH,
	AND,
	OR,
	XOR,
	NOT,
	LE,
	LT,
	GE,
	GT,
	EQ,
	NE
};
class expression
{
public:
	virtual ~expression() {}
	virtual float evaluate(const driver& drv) const = 0;
	virtual float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const =0;
	virtual void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const = 0;
};

class unary_expression : public expression
{
public:
	unary_expression(operation op, expr_ptr expr);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	operation op;
	expr_ptr expr;
};

class binary_expression : public expression
{
public:
	binary_expression(operation op, expr_ptr left, expr_ptr right);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	operation op;
	expr_ptr left;
	expr_ptr right;
};

class ternary_expression : public expression
{
public:
	ternary_expression(expr_ptr left, expr_ptr middle, expr_ptr right);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	expr_ptr left;
	expr_ptr middle;
	expr_ptr right;
};

class parenthesis_expression : public expression
{
public:
	parenthesis_expression(expr_ptr expr);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	expr_ptr expr;
};

class literal_expression : public expression
{
public:
	literal_expression(float value);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	float value;
};

class identifier_expression : public expression
{
public:
	identifier_expression(std::string name);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	std::string name;
};

class variable_expression : public expression
{
public:
	variable_expression(std::string name);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	std::string name;
};

class alias_expression : public expression
{
public:
	alias_expression(std::string name);
	float evaluate(const driver& drv) const override;
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	std::string name;
};


//only for use in expressions in the upp file
class p_expression : public expression
{
public:
	p_expression(std::set<std::pair<std::string,int>> node_state_pairs);
	
	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;
	float evaluate(const driver& drv) const override;

	void generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const override;

	std::set<std::pair<std::string,int>> node_state_pairs;
};



class external_input_expression :public expression
{
public:
	external_input_expression(std::string name);

	float evaluate(const driver &drv) const override;
	

	float evaluate(const driver& drv,std::unordered_map<std::set<std::pair<std::string, int>>, float, set_hash> p_inputs) const override;

	void generate_code(const driver &drv,const std::string& current_node,std::ostream& os) const override;
	std::string name;
	expr_ptr expr;
};
