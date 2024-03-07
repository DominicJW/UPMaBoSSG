#include "expressions.h"

#include <algorithm>
#include <stdexcept>
#include <cmath>
#include <numeric>

#include "driver.h"
#include "../state_word.h"
#define DIV_UP(x, y) ((x) + (y)-1) / (y)


unary_expression::unary_expression(operation op, expr_ptr expr) : op(op), expr(std::move(expr)) {}

float unary_expression::evaluate(const driver& drv) const
{
	switch (op)
	{
		case operation::PLUS:
			return expr->evaluate(drv);
		case operation::MINUS:
			return -expr->evaluate(drv);
		case operation::NOT:
			return !expr->evaluate(drv);
		default:
			throw std::runtime_error("Unknown unary operator");
	}
}

void unary_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	switch (op)
	{
		case operation::PLUS:
			os << "+";
			expr->generate_code(drv, current_node, os);
			break;
		case operation::MINUS:
			os << "-";
			expr->generate_code(drv, current_node, os);
			break;
		case operation::NOT:
			os << "!";
			expr->generate_code(drv, current_node, os);
			break;
		default:
			throw std::runtime_error("Unknown unary operator");
	}
}

binary_expression::binary_expression(operation op, expr_ptr left, expr_ptr right)
	: op(op), left(std::move(left)), right(std::move(right))
{}

float binary_expression::evaluate(const driver& drv) const
{
	switch (op)
	{
		case operation::PLUS:
			return left->evaluate(drv) + right->evaluate(drv);
		case operation::MINUS:
			return left->evaluate(drv) - right->evaluate(drv);
		case operation::STAR:
			return left->evaluate(drv) * right->evaluate(drv);
		case operation::SLASH:
			return left->evaluate(drv) / right->evaluate(drv);
		case operation::AND:
			return left->evaluate(drv) && right->evaluate(drv);
		case operation::OR:
			return left->evaluate(drv) || right->evaluate(drv);
		case operation::EQ:
			return left->evaluate(drv) == right->evaluate(drv);
		case operation::NE:
			return left->evaluate(drv) != right->evaluate(drv);
		case operation::LE:
			return left->evaluate(drv) <= right->evaluate(drv);
		case operation::LT:
			return left->evaluate(drv) < right->evaluate(drv);
		case operation::GE:
			return left->evaluate(drv) >= right->evaluate(drv);
		case operation::GT:
			return left->evaluate(drv) > right->evaluate(drv);
		default:
			throw std::runtime_error("Unknown binary operator " + std::to_string(static_cast<int>(op)));
	}
}

void binary_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	switch (op)
	{
		case operation::PLUS:
			left->generate_code(drv, current_node, os);
			os << " + ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::MINUS:
			left->generate_code(drv, current_node, os);
			os << " - ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::STAR:
			left->generate_code(drv, current_node, os);
			os << " * ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::SLASH:
			left->generate_code(drv, current_node, os);
			os << " / ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::AND:
			left->generate_code(drv, current_node, os);
			os << " & ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::OR:
			left->generate_code(drv, current_node, os);
			os << " | ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::EQ:
			left->generate_code(drv, current_node, os);
			os << " == ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::NE:
			left->generate_code(drv, current_node, os);
			os << " != ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::LE:
			left->generate_code(drv, current_node, os);
			os << " <= ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::LT:
			left->generate_code(drv, current_node, os);
			os << " < ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::GE:
			left->generate_code(drv, current_node, os);
			os << " >= ";
			right->generate_code(drv, current_node, os);
			break;
		case operation::GT:
			left->generate_code(drv, current_node, os);
			os << " > ";
			right->generate_code(drv, current_node, os);
			break;
		default:
			throw std::runtime_error("Unknown binary operator " + std::to_string(static_cast<int>(op)));
	}
}

ternary_expression::ternary_expression(expr_ptr left, expr_ptr middle, expr_ptr right)
	: left(std::move(left)), middle(std::move(middle)), right(std::move(right))
{}

float ternary_expression::evaluate(const driver& drv) const
{
	return left->evaluate(drv) ? middle->evaluate(drv) : right->evaluate(drv);
}

void ternary_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	left->generate_code(drv, current_node, os);
	os << " ? ";
	middle->generate_code(drv, current_node, os);
	os << " : ";
	right->generate_code(drv, current_node, os);
}

parenthesis_expression::parenthesis_expression(expr_ptr expr) : expr(std::move(expr)) {}

float parenthesis_expression::evaluate(const driver& drv) const { return expr->evaluate(drv); }

void parenthesis_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	os << "(";
	expr->generate_code(drv, current_node, os);
	os << ")";
}

literal_expression::literal_expression(float value) : value(value) {}

float literal_expression::evaluate(const driver&) const { return value; }

void literal_expression::generate_code(const driver&, const std::string&, std::ostream& os) const { os << value; }

identifier_expression::identifier_expression(std::string name) : name(std::move(name)) {}

float identifier_expression::evaluate(const driver&) const
{
	throw std::runtime_error("identifier " + name + "in expression which needs to be evaluated");
}

void identifier_expression::generate_code(const driver& drv, const std::string&, std::ostream& os) const
{
	auto it = std::find_if(drv.nodes.begin(), drv.nodes.end(), [this](auto&& node) { return node.name == name; });
	if (it == drv.nodes.end())
	{
		throw std::runtime_error("unknown node name: " + name);
	}
	int i = it - drv.nodes.begin();
	int word = i / 32;
	int bit = i % 32;
	os << "((state[" << word << "] & " << (1u << bit) << "u) != 0)";
}

variable_expression::variable_expression(std::string name) : name(std::move(name)) {}

float variable_expression::evaluate(const driver& drv) const { return drv.variables.at(name); }

void variable_expression::generate_code(const driver& drv, const std::string&, std::ostream& os) const
{
	os << drv.variables.at(name);
}

alias_expression::alias_expression(std::string name) : name(std::move(name)) {}

float alias_expression::evaluate(const driver&) const
{
	throw std::runtime_error("alias " + name + "in expression which needs to be evaluated");
}

void alias_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	auto it = std::find_if(drv.nodes.begin(), drv.nodes.end(), [&](auto&& node) { return node.name == current_node; });
	assert(it != drv.nodes.end());

	auto&& attr = it->get_attr(name.substr(1));

	attr.second->generate_code(drv, current_node, os);
}




external_input_expression::external_input_expression(std::string name) : name(std::move(name)) {};

//not to be called
float external_input_expression::evaluate(const driver& drv) const
{
	throw std::runtime_error("extenal_inp " + name + "in expression which needs to be evaluated");
}

//for generating the kernel code
void external_input_expression::generate_code(const driver &drv,const std::string& current_node,std::ostream& os) const
{


	auto it = std::find_if(drv.external_inputs.begin(), drv.external_inputs.end(), [&](auto&& ext_inp) { return ext_inp.name == name;});
	if (it == drv.external_inputs.end()) {
		throw std::runtime_error("unknown node name: ");
	}

	int ext_inp_idx = std::distance(drv.external_inputs.begin(), it);

	os << "external_inputs[" << std::to_string(ext_inp_idx)<<"]";
}

float external_input_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const
{
	throw std::runtime_error("extenal_inp " + name + "in expression which needs to be evaluated");
}



p_expression::p_expression(std::vector<std::string> node_name_list, std::vector<int> state_list) : node_name_list(node_name_list), state_list(state_list)
{

}

float p_expression::evaluate(const driver& drv,std::vector<state_word_t> last_states) const
{
	//although unchanging for each time evaluate is called, the complexity overhead is negligible. 
	int state_words = DIV_UP(drv.nodes.size(), 32);
	float total = 0;
	std::vector<state_word_t> on_states(state_words,0);
	std::vector<state_word_t> off_states(state_words,0);
	for (int i = 0; i< node_name_list.size(); i++)
	{
		auto it = std::find_if(drv.nodes.begin(), drv.nodes.end(), [&](auto&& node) { return node.name == node_name_list[i]; });
		if (it == drv.nodes.end()) {
			
		    throw std::runtime_error("unknown node name: ");
		}

		int node_idx = std::distance(drv.nodes.begin(), it);
		int word = node_idx / 32;
		int bit = node_idx % 32;
		if (state_list[i])
		{
			on_states[word] = on_states[word] + std::pow(2,bit);
		}
		else
		{
			off_states[word] = off_states[word] + std::pow(2,bit);
		}
	}
	//should be 'cells in simulation right now' variable not this
	//calculate by dividing last_states size by state_words
	//last_states size steadily increasing
	float number_of_cells = DIV_UP(last_states.size(),state_words);
	for (int cell_idx = 0;cell_idx<  number_of_cells; cell_idx++)
	{ 
    	std::vector<state_word_t> cell_last_states(last_states.begin() + (cell_idx * state_words),last_states.begin() + (cell_idx * state_words)+state_words);
		std::vector<state_word_t> on_and(state_words);
		std::vector<state_word_t> off_and(state_words);

    	std::transform(on_states.begin(), on_states.end(), cell_last_states.begin(), on_and.begin(), std::bit_and<state_word_t>());
    	std::transform(off_states.begin(), off_states.end(), cell_last_states.begin(), off_and.begin(), std::bit_and<state_word_t>());		
		bool condition_true_for_sample = std::equal(on_states.begin(), on_states.end(), on_and.begin(), on_and.end()) && (std::accumulate(off_and.begin(), off_and.end(), 0) == 0) ;
		total = total + condition_true_for_sample;
	}
	return total/number_of_cells;
}

void p_expression::generate_code(const driver& drv, const std::string& current_node, std::ostream& os) const
{
	throw std::runtime_error("attempting to generate code for p_exp in device code");

}

float p_expression::evaluate(const driver& drv) const
{
	throw "cannot evaluate p_expression without states.";
} 



float unary_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const
{
	switch (op)
	{
		case operation::PLUS:
			return expr->evaluate(drv,last_states);
		case operation::MINUS:
			return -expr->evaluate(drv,last_states);
		case operation::NOT:
			return !expr->evaluate(drv,last_states);
		default:
			throw std::runtime_error("Unknown unary operator");
	}
}


float binary_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const
{
	switch (op)
	{
		case operation::PLUS:
			return left->evaluate(drv,last_states) + right->evaluate(drv,last_states);
		case operation::MINUS:
			return left->evaluate(drv,last_states) - right->evaluate(drv,last_states);
		case operation::STAR:
			return left->evaluate(drv,last_states) * right->evaluate(drv,last_states);
		case operation::SLASH:
			return left->evaluate(drv,last_states) / right->evaluate(drv,last_states);
		case operation::AND:
			return left->evaluate(drv,last_states) && right->evaluate(drv,last_states);
		case operation::OR:
			return left->evaluate(drv,last_states) || right->evaluate(drv,last_states);
		case operation::EQ:
			return left->evaluate(drv,last_states) == right->evaluate(drv,last_states);
		case operation::NE:
			return left->evaluate(drv,last_states) != right->evaluate(drv,last_states);
		case operation::LE:
			return left->evaluate(drv,last_states) <= right->evaluate(drv,last_states);
		case operation::LT:
			return left->evaluate(drv,last_states) < right->evaluate(drv,last_states);
		case operation::GE:
			return left->evaluate(drv,last_states) >= right->evaluate(drv,last_states);
		case operation::GT:
			return left->evaluate(drv,last_states) > right->evaluate(drv,last_states);
		default:
			throw std::runtime_error("Unknown binary operator " + std::to_string(static_cast<int>(op)));
	}
}

float ternary_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const
{
	return left->evaluate(drv,last_states) ? middle->evaluate(drv,last_states) : right->evaluate(drv,last_states);
}


float parenthesis_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const { return expr->evaluate(drv,last_states); }

float literal_expression::evaluate(const driver&,std::vector<state_word_t> last_states) const { return value; }


float identifier_expression::evaluate(const driver& drv,std::vector<state_word_t> last_states) const
{
	throw std::runtime_error("identifier " + name + "in expression which needs to be evaluated");
}

float variable_expression::evaluate(const driver& drv, std::vector<state_word_t> last_states ) const { return drv.variables.at(name); }


float alias_expression::evaluate(const driver& drv,std::vector<state_word_t> last_states) const
{
	throw std::runtime_error("alias " + name + "in expression which needs to be evaluated");
}
