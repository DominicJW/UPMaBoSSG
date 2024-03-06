#pragma once

#include <vector>

#include "kernel.h"
#include "statistics/stats_composite.h"
#include "./parser/driver.h"

class simulation_runner
{
	int n_trajectories_;
	int state_size_;
	int state_words_;
	unsigned long long seed_;
	std::vector<float> inital_probs_;
	driver& drv;

public:
	int trajectory_len_limit;
	int trajectory_batch_limit;

	simulation_runner(int n_trajectories, int state_size, unsigned long long seed, std::vector<float> inital_probs,driver& drv);

	void run_simulation(stats_composite& stats_runner, kernel_wrapper& initialize_random,
						kernel_wrapper& initialize_initial_state, kernel_wrapper& simulate);
};
