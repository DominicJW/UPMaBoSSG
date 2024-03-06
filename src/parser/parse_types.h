#pragma once

#include <algorithm>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

#include "expressions.h"

using node_attr_t = std::pair<std::string, expr_ptr>;
using node_attr_list_t = std::vector<node_attr_t>;

class driver;

struct node_t
{
	std::string name;
	node_attr_list_t attrs;
	float istate;

	node_t(std::string name, node_attr_list_t attrs);

	const node_attr_t& get_attr(std::string_view name) const;

	bool has_attr(std::string_view name) const;

	bool is_internal(const driver& drv) const;
};

struct ext_inp_t
{
	std::string name;
	expr_ptr expr;

	ext_inp_t(std::string name, expr_ptr expr);
};
