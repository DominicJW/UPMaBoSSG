#pragma once

#include <thrust/host_vector.h>

#include <curand_kernel.h>
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

	thrust::host_vector<state_word_t> saved_states;
	thrust::host_vector<float> saved_times;
	thrust::host_vector<curandState> saved_rands;

public:
	int trajectory_len_limit;
	int trajectory_batch_limit;

	simulation_runner(int n_trajectories, int state_size, unsigned long long seed, std::vector<float> inital_probs,driver& drv);

	void run_simulation(stats_composite& stats_runner, kernel_wrapper& initialize_random,
						kernel_wrapper& initialize_initial_state, kernel_wrapper& simulate,const std::string& output_prefix);

	void save_trajs_before_overwrite(int trajectories_in_batch,int new_batch_addition,thrust::device_ptr<state_word_t> d_last_states,thrust::device_ptr<float> d_last_times, thrust::device_ptr<curandState> d_rands);

	void load_batch_addition_from_saved_and_new(int trajectories_in_batch,int new_batch_addition,thrust::device_ptr<state_word_t> d_last_states,thrust::device_ptr<float> d_last_times, thrust::device_ptr<curandState> d_rands);
};
