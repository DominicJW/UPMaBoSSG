#pragma once

#include <map>

#include "../kernel.h"
#include "../state.h"
#include "stats.h"

class final_states_stats : public stats
{
	std::vector<int> result_occurences_;
	thrust::device_ptr<int> occurences_;

	int noninternal_states_count_;
	state_t noninternals_mask_;

	kernel_wrapper& final_states_;

public:
	final_states_stats(state_t noninternals_mask_, int noninternals, kernel_wrapper& final_states);
	~final_states_stats();

	void process_batch_internal(thrust::device_ptr<state_word_t> last_states,
								thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories_batch);

	void process_batch(thrust::device_ptr<state_word_t> traj_states, thrust::device_ptr<float> traj_times,
					   thrust::device_ptr<float> traj_tr_entropies, thrust::device_ptr<state_word_t> last_states,
					   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) override;

	void finalize() override;
	void reset() override;

	void visualize(int n_trajectories, const std::vector<std::string>& nodes) override;
	void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string& prefix) override;
};
