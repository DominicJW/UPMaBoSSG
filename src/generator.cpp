#include "generator.h"

#include <algorithm>
#include <sstream>

#include "timer.h"
#include "utils.h"

generator::generator(driver& drv) : drv_(drv) {}

std::string generator::generate_code() const
{
	timer_stats stats("generator> generate");

	std::ostringstream ss;

	ss << "using uint8_t = unsigned char;" << std::endl;
	ss << "using uint32_t = unsigned int;" << std::endl << std::endl;

	ss << "constexpr int state_size = " << drv_.nodes.size() << ";" << std::endl;
	ss << "constexpr int state_words = " << DIV_UP(drv_.nodes.size(), 32) << ";" << std::endl;
	ss << "constexpr bool discrete_time = " << (drv_.constants["discrete_time"] != 0) << ";" << std::endl;
	ss << "constexpr float max_time = " << drv_.constants["max_time"] << ";" << std::endl;
	ss << "constexpr float time_tick = " << drv_.constants["time_tick"] << ";" << std::endl;

	const char* state_cuh =
#include "jit_kernels/include/state_word.h"
		;
	ss << state_cuh << std::endl;

	generate_node_transitions(ss);
	ss << std::endl;

	generate_aggregate_function(ss);
	ss << std::endl;

	generate_transition_entropy_function(ss);
	ss << std::endl;

	const char* traj_status_cu =
#include "jit_kernels/include/trajectory_status.h"
		;
	ss << traj_status_cu << std::endl;

	generate_simulate(ss);
	ss << std::endl;

	generate_non_internal_index(ss);
	ss << std::endl;

	if (timer_stats::enable_diags())
		std::cerr << ss.str() << std::endl;

	return ss.str();
}

bool is_logic_ternary_expression(const driver& drv, const expression* expr, bool up, float& val)
{
	const ternary_expression* ternary_expr = dynamic_cast<const ternary_expression*>(expr);

	if (!ternary_expr)
		return false;

	const alias_expression* alias_expr = dynamic_cast<const alias_expression*>(ternary_expr->left.get());

	if (!alias_expr)
		return false;

	if (alias_expr->name != "@logic")
		return false;

	auto l_val = ternary_expr->middle->evaluate(drv);
	auto r_val = ternary_expr->right->evaluate(drv);

	auto zero = up ? r_val : l_val;
	val = up ? l_val : r_val;

	if (zero != 0.f)
		return false;

	return true;
}

void generator::generate_node_transitions(std::ostringstream& os) const
{
	for (auto&& node : drv_.nodes)
	{
		os << "__device__ float " << node.name << "_rate(const state_word_t* __restrict__ state,float* external_inputs) " << std::endl;
		os << "{" << std::endl;

		float up_val, down_val;

		if (is_logic_ternary_expression(drv_, node.get_attr("rate_up").second.get(), true, up_val)
			&& is_logic_ternary_expression(drv_, node.get_attr("rate_down").second.get(), false, down_val))
		{
			os << "    const bool is_up = ";
			identifier_expression(node.name).generate_code(drv_, node.name, os);
			os << ";" << std::endl;
			os << "    const bool logic = ";
			node.get_attr("logic").second->generate_code(drv_, node.name, os);
			os << ";" << std::endl;

			if (up_val == down_val)
				os << "    return is_up == logic ? 0 : " << up_val << ";" << std::endl;
			else
				os << "    return is_up == logic ? 0 : (is_up ? " << down_val << " : " << up_val << ");" << std::endl;
		}
		else
		{
			os << "    return ";
			identifier_expression(node.name).generate_code(drv_, node.name, os);
			os << " ?" << std::endl;
			os << "          (";
			node.get_attr("rate_down").second->generate_code(drv_, node.name, os);
			os << ")" << std::endl;
			os << "        : (";
			node.get_attr("rate_up").second->generate_code(drv_, node.name, os);
			os << ");" << std::endl;
		}
		os << "}" << std::endl << std::endl;
	}
}

void generator::generate_aggregate_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_rates(float* __restrict__ transition_rates, const state_word_t* "
		  "__restrict__ state,float* external_inputs)"
	   << std::endl;
	os << "{" << std::endl;
	os << "    float sum = 0;" << std::endl;
	os << "    float tmp;" << std::endl;
	os << std::endl;

	int i = 0;
	for (auto&& node : drv_.nodes)
	{
		os << "    tmp = " << node.name << "_rate(state,external_inputs);" << std::endl;
		os << "    transition_rates[" << i++ << "] = tmp;" << std::endl;
		os << "    sum += tmp;" << std::endl;
		os << std::endl;
	}
	os << "    return sum;" << std::endl;
	os << "}" << std::endl;
}

void generator::generate_transition_entropy_function(std::ostringstream& os) const
{
	os << "__device__ float compute_transition_entropy(const float* __restrict__ transition_rates)" << std::endl;
	os << "{" << std::endl;
	os << "    float entropy = 0.f;" << std::endl;
	os << "    float non_internal_total_rate = 0.f;" << std::endl;
	os << "    float tmp_prob;" << std::endl << std::endl;

	for (size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!drv_.nodes[i].is_internal(drv_))
		{
			os << "    non_internal_total_rate += transition_rates[" << i << "];" << std::endl;
		}
	}

	os << "    if (non_internal_total_rate == 0.f) return 0.f;" << std::endl << std::endl;

	for (size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!drv_.nodes[i].is_internal(drv_))
		{
			os << "    tmp_prob = transition_rates[" << i << "] / non_internal_total_rate;" << std::endl;
			os << "    entropy -= (tmp_prob == 0.f) ? 0.f : log2f(tmp_prob) * tmp_prob;" << std::endl;
		}
	}

	os << "    return entropy;" << std::endl;
	os << "}";
}

void generator::generate_simulate(std::ostringstream& os) const
{
	os << R"(
extern __device__ void simulate_inner(int trajectories_count, int state_size, int trajectory_limit, float time_tick,
									  float max_time, bool discrete_time, state_word_t* __restrict__ last_states,
									  float* __restrict__ last_times, void* __restrict__ rands,
									  state_word_t* __restrict__ trajectory_states,
									  float* __restrict__ trajectory_times,
									  float* __restrict__ trajectory_transition_entropies,
									  trajectory_status* __restrict__ trajectory_statuses,
									  float* __restrict__ transition_rates, state_word_t* __restrict__ state,float* external_inputs);

extern "C" __global__ void simulate(int trajectories_count, int trajectory_limit,
									state_word_t* __restrict__ last_states, float* __restrict__ last_times,
									void* __restrict__ rands, state_word_t* __restrict__ trajectory_states,
									float* __restrict__ trajectory_times,
									float* __restrict__ trajectory_transition_entropies,
									trajectory_status* __restrict__ trajectory_statuses,float* external_inputs)
{
	float transition_rates[state_size];

	state_word_t state[state_words];

	simulate_inner(trajectories_count, state_size, trajectory_limit, time_tick, max_time, discrete_time, last_states,
				   last_times, rands, trajectory_states, trajectory_times, trajectory_transition_entropies,
				   trajectory_statuses, transition_rates, state,external_inputs);
}
)";
}

void generator::generate_non_internal_index(std::ostringstream& os) const
{
	os << "__device__ uint32_t get_non_internal_index(const state_word_t* __restrict__ state)" << std::endl;
	os << "{" << std::endl;
	os << "    return" << std::endl;
	os << "			";
	int non_internals_count =
		std::count_if(drv_.nodes.begin(), drv_.nodes.end(), [&](const auto& node) { return !node.is_internal(drv_); });
	int non_internals = 0;
	for (size_t i = 0; i < drv_.nodes.size(); i++)
	{
		if (!drv_.nodes[i].is_internal(drv_))
		{
			os << "((state[" << i / 32 << "] & " << (1u << i % 32) << "u)";
			auto shift = (int)(i % 32) - non_internals++;
			if (shift > 0)
				os << " >> " << shift;
			else if (shift < 0)
				os << " << " << -shift;
			os << ")";

			if (non_internals != non_internals_count)
			{
				os << " | ";
			}
		}
	}
	os << ";" << std::endl << "}" << std::endl;
}



