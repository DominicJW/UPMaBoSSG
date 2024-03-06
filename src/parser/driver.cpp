#include "driver.h"

#include <algorithm>

#include "../timer.h"

driver::driver() : trace_parsing(false), trace_scanning(false), start(start_type::none)
{
	constants["discrete_time"] = 0;
	constants["sample_count"] = 1'000'000;
	constants["max_time"] = 10;
	constants["time_tick"] = 0.1;
	constants["seed_pseudorandom"] = 0;
}


int driver::parse(std::string bnd_file, std::string cfg_file,std::string upp_file)
{

	int res;
	{
		timer_stats t("parser> parse_bnd");
		start = start_type::bnd;
		res = parse_one(bnd_file);
		if (res != 0)
			return res;
	}

	{
		timer_stats t("parser> parse_cfg");
		start = start_type::cfg;
		res = parse_one(cfg_file);
		if (res != 0)
			return res;
	}
	
	{
		if(upp_file != "NOFILE")
		{
			timer_stats t("parser> parse_upp");
			start = start_type::upp;
			return parse_one(upp_file);
		}
	}
	return 0;
}



int driver::parse_one(std::string f)
{
	

	file = std::move(f);
	location.initialize(&file);
	scan_begin();
	yy::parser parse(*this);
	parse.set_debug_level(trace_parsing);
	int res = parse();
	scan_end();
	return res;
}

void driver::register_variable(std::string name, expr_ptr expr) { variables[std::move(name)] = expr->evaluate(*this); }

void driver::register_constant(std::string name, expr_ptr expr) { constants[std::move(name)] = expr->evaluate(*this); }

void driver::register_node(std::string name, node_attr_list_t node_attrs)
{
	node_t node(std::move(name), std::move(node_attrs));

	if (std::find_if(nodes.begin(), nodes.end(), [&](auto&& n) { return n.name == node.name; }) != nodes.end())
		throw std::runtime_error("Node " + name + " already exists");

	if (!node.has_attr("rate_up"))
		throw std::runtime_error("Node " + name + " does not have rate_up attribute");

	if (!node.has_attr("rate_down"))
		throw std::runtime_error("Node " + name + " does not have rate_down attribute");

	nodes.emplace_back(std::move(node));
}

void driver::register_node_attribute(std::string node_name, std::string attr_name, expr_ptr expr)
{
	if (auto it = std::find_if(nodes.begin(), nodes.end(), [&](auto&& node) { return node.name == node_name; });
		it != nodes.end())
	{
		if (attr_name == "istate")
			it->istate = expr->evaluate(*this);
		else
			it->attrs.emplace_back(std::move(attr_name), std::move(expr));
	}
	else
		throw std::runtime_error("Unknown node " + node_name);
}

void driver::register_node_istate(std::string node_name, expr_ptr expr_l, expr_ptr expr_r, int value_l)
{
	if (auto it = std::find_if(nodes.begin(), nodes.end(), [&](auto&& node) { return node.name == node_name; });
		it != nodes.end())
	{
		auto l_prob = expr_l->evaluate(*this);
		auto r_prob = expr_r->evaluate(*this);
		float activate_prob;
		if (value_l == 1)
		{
			activate_prob = l_prob / (l_prob + r_prob);
		}
		else
		{
			activate_prob = r_prob / (l_prob + r_prob);
		}
		it->istate = activate_prob;
	}
	else
		throw std::runtime_error("Unknown node " + node_name);
}




void driver::register_external_input(std::string name)
{
	throw "this constructor not implemented";
	// ext_inp_t ext_inp_t(std::move(name)); //remember c++ funny constructor

	// external_inputs.emplace_back(ext_inp);
}

void driver::register_external_input(std::string name,expr_ptr expr)
{
	ext_inp_t myvar(std::move(name), std::move(expr)); //remember c++ funny constructor

	external_inputs.emplace_back(std::move(myvar));
}


void driver::register_external_input_expression_attribute(ext_inp_t ext_inp,expr_ptr expr)
{
	
	throw "register_external_input_expression_attribute not implemented";
}

void driver::register_external_input_expression_attribute(std::string name,expr_ptr expr)
{
	throw "register_external_input_expression_attribute not implemented";

}