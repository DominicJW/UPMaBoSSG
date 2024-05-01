#include "stats_composite.h"

void stats_composite::add(stats_ptr&& stat) { composed_stats_.emplace_back(std::move(stat)); }

void stats_composite::process_batch(thrust::device_ptr<state_word_t> traj_states, thrust::device_ptr<float> traj_times,
									thrust::device_ptr<float> traj_tr_entropies,
									thrust::device_ptr<state_word_t> last_states,
									thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories)
{
	for (auto&& stat : composed_stats_)
		stat->process_batch(traj_states, traj_times, traj_tr_entropies, last_states, traj_statuses, n_trajectories);
}

void stats_composite::finalize()
{
	for (auto&& stat : composed_stats_)
		stat->finalize();
}


void stats_composite::reset()
{
	for (auto&& stat : composed_stats_)
		stat->reset();
}

void stats_composite::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	for (auto&& stat : composed_stats_)
		stat->visualize(n_trajectories, nodes);
}

void stats_composite::write_csv(int n_trajectories, const std::vector<std::string>& nodes, const std::string& prefix)
{
	for (auto&& stat : composed_stats_)
		stat->write_csv(n_trajectories, nodes, prefix);
}
