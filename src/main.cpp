#include <fstream>
#include <iostream>
#include <optional>

#include <vector>

#include "generator.h"
#include "kernel_compiler.h"
#include "simulation_runner.h"
#include "state_word.h"
#include "statistics/final_states.h"
#include "statistics/fixed_states.h"
#include "statistics/stats_composite.h"
#include "statistics/window_average_small.h"
#include "timer.h"




state_t create_noninternals_mask(driver& drv)
{
	state_t mask(drv.nodes.size());
	for (size_t i = 0; i < drv.nodes.size(); ++i)
	{
		if (!drv.nodes[i].is_internal(drv))
			mask.set(i);
	}

	return mask;
}

std::vector<float> create_initial_probs(driver& drv)
{
	std::vector<float> initial_probs;
	for (auto&& node : drv.nodes)
	{
		initial_probs.push_back(node.istate);
	}

	return initial_probs;
}

int do_compilation(driver& drv, bool discrete_time, std::optional<kernel_compiler>& compiler)
{
	timer_stats stats("main> compilation");

	generator gen(drv);

	auto s = gen.generate_code();

	std::string filename = "generated/generated_code.cpp";
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error creating file: " << filename << std::endl;
        return 1; // Return an error code
    }
    std::string content = s;
    file << content;
    file.close();
    std::cout << "File written successfully: " << filename << std::endl;
	std::ifstream read_file("generated/generated_code.cpp");
    std::string fileContents;
    if (read_file) {
        std::stringstream buffer;
        buffer << read_file.rdbuf();
        fileContents = buffer.str();
    }


	compiler.emplace();

	if (compiler->compile_simulation(fileContents, discrete_time))
		return 1;

	return 0;
}

stats_composite do_simulation(bool discrete_time, float max_time, float time_tick, int sample_count, int state_size,
							  unsigned long long seed, std::vector<float> initial_probs,
							  const state_t& noninternals_mask, int noninternals_count, kernel_compiler& compiler,driver& drv,const std::string& output_prefix)
{
	timer_stats stats("main> simulation");

	simulation_runner r(sample_count, state_size, seed, std::move(initial_probs),drv);

	stats_composite stats_runner;

	// for final states
	stats_runner.add(
		std::make_unique<final_states_stats>(noninternals_mask, noninternals_count, compiler.final_states));

	// for fixed states
	fixed_states_stats_builder::add_fixed_states_stats(stats_runner, noninternals_mask.words_n());

	// for window averages
	stats_runner.add(std::make_unique<window_average_small_stats>(
		time_tick, max_time, discrete_time, noninternals_mask, noninternals_count, r.trajectory_len_limit,
		r.trajectory_batch_limit, compiler.window_average_small));

	// // run
	r.run_simulation(stats_runner, compiler.initialize_random, compiler.initialize_initial_state, compiler.simulate,output_prefix);

	// // finalize
	
	// stats_runner.finalize();

	return stats_runner;
}

void do_visualization(stats_composite& stats_runner, int sample_count, const std::vector<std::string>& node_names,
					  const std::string& output_prefix)
{
	timer_stats stats("main> visualization");

	// visualize
	if (output_prefix.size() > 0)
	{
		stats_runner.write_csv(sample_count, node_names, output_prefix);
	}
	else
	{

		stats_runner.write_csv(sample_count, node_names, "no-prefix");
		// stats_runner.visualize(sample_count, node_names);
	}
}

int main(int argc, char** argv)
{


	std::vector<std::string> args(argv + 1, argv + argc);


	std::string output_prefix = "";
	std::string bnd_path = args[0];
	std::string cfg_path = args[1];
	std::string upp_path = "NOFILE";
	if (args[0] == "-o")
	{
		output_prefix = args[1];
		bnd_path = args[2];
		cfg_path = args[3];
		if (args.size() == 5)
			upp_path = args[4];
	}
	else if (args.size() == 3)
			upp_path = args[2];



	driver drv;
	{
		timer_stats stats("main> compilation");
		if (drv.parse(bnd_path, cfg_path,upp_path))
			return 1;
	}

	bool discrete_time = drv.constants["discrete_time"] != 0;
	float max_time = drv.constants["max_time"];
	float time_tick = drv.constants["time_tick"];
	int sample_count = drv.constants["sample_count"];
	unsigned long long seed = drv.constants["seed_pseudorandom"];
	auto initial_probs = create_initial_probs(drv);
	auto noninternals_mask = create_noninternals_mask(drv);
	int noninternals_count =
		std::count_if(drv.nodes.begin(), drv.nodes.end(), [&](const auto& node) { return !node.is_internal(drv); });

	std::vector<std::string> node_names;
	for (auto&& node : drv.nodes)
		node_names.push_back(node.name);

	if (noninternals_count > 20)
	{
		std::cerr << "This executable supports a maximum of 20 non-internal nodes." << std::endl;
		return 1;
	}

	if (drv.nodes.size() > MAX_NODES)
	{
		std::cerr << "This executable supports a maximum of " << MAX_NODES << " nodes." << std::endl;
		return 1;
	}


	{
		std::optional<kernel_compiler> compiler;

		if (do_compilation(drv, discrete_time, compiler))
			return 1;

		auto stats_runner = do_simulation(discrete_time, max_time, time_tick, sample_count, drv.nodes.size(), seed,
										  std::move(initial_probs), noninternals_mask, noninternals_count, *compiler,drv,output_prefix);

		// do_visualization(stats_runner, sample_count, node_names, output_prefix);
	}

	timer_stats::print_aggregate_stats();

	return 0;
}
