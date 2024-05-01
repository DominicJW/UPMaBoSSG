#pragma once

#include <string>
#include <vector>

#include <thrust/device_ptr.h>

#include "../state_word.h"
#include "../trajectory_status.h"

class stats;

using stats_ptr = std::unique_ptr<stats>;

class stats
{
public:
	virtual ~stats() = default;

	virtual void process_batch(thrust::device_ptr<state_word_t> traj_states, thrust::device_ptr<float> traj_times,
							   thrust::device_ptr<float> traj_tr_entropies,
							   thrust::device_ptr<state_word_t> last_states,
							   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories) = 0;

	virtual void finalize() {}
	virtual void reset () {}

	virtual void visualize(int n_trajectories, const std::vector<std::string>& nodes) = 0;
	virtual void write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string& prefix) = 0;
};
