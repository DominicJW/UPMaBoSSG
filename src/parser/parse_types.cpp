#include "parse_types.h"

node_t::node_t(std::string name, node_attr_list_t attrs) : name(std::move(name)), attrs(std::move(attrs)), istate(0.5)
{}

const node_attr_t& node_t::get_attr(std::string_view name) const
{
	if (auto it = std::find_if(attrs.begin(), attrs.end(), [&](auto&& attr) { return attr.first == name; });
		it != attrs.end())
		return *it;
	else
		throw std::runtime_error("Node " + this->name + " does not have attribute " + std::string(name));
}

bool node_t::has_attr(std::string_view name) const
{
	return std::find_if(attrs.begin(), attrs.end(), [&](auto&& attr) { return attr.first == name; }) != attrs.end();
}

bool node_t::is_internal(const driver& drv) const
{
	return has_attr("is_internal") && get_attr("is_internal").second->evaluate(drv) != 0;
}

ext_inp_t::ext_inp_t(std::string name, expr_ptr expr) : name(std::move(name)), expr(std::move(expr)) {}