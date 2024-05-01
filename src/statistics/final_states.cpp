#include "final_states.h"

#include <fstream>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

#include "../timer.h"
#include "window_average_small.h"

final_states_stats::final_states_stats(state_t noninternals_mask, int noninternals, kernel_wrapper& final_states)
	: noninternal_states_count_(1 << noninternals),
	  noninternals_mask_(std::move(noninternals_mask)),
	  final_states_(final_states)
{
	timer_stats stats("final_states_stats> initialize");

	occurences_ = thrust::device_malloc<int>(noninternal_states_count_);
	result_occurences_.resize(noninternal_states_count_);
}

final_states_stats::~final_states_stats()
{
	timer_stats stats("final_states_stats> free");

	thrust::device_free(occurences_);
}

void final_states_stats::process_batch(thrust::device_ptr<state_word_t>, thrust::device_ptr<float>,
									   thrust::device_ptr<float>, thrust::device_ptr<state_word_t> last_states,
									   thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories)
{
	process_batch_internal(last_states, traj_statuses, n_trajectories);
}

void final_states_stats::process_batch_internal(thrust::device_ptr<state_word_t> last_states,
												thrust::device_ptr<trajectory_status> traj_statuses, int n_trajectories)
{
	timer_stats stats("final_states_stats> process_batch");

	final_states_.run(DIV_UP(n_trajectories, 256), 256, n_trajectories, (int)noninternals_mask_.words_n(),
					  last_states.get(), traj_statuses.get(), occurences_.get());
}

void final_states_stats::finalize()
{
	timer_stats stats("final_states_stats> finalize");

	CUDA_CHECK(cudaMemcpy(result_occurences_.data(), occurences_.get(), noninternal_states_count_ * sizeof(int),
						  cudaMemcpyDeviceToHost));
}

void final_states_stats::reset()
{
	timer_stats stats("final_states_stats> finalize");

	CUDA_CHECK(cudaMemset(occurences_.get(), 0, noninternal_states_count_ * sizeof(int)));
}


void final_states_stats::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	timer_stats stats("final_states_stats> visualize");

	std::cout << "final points:" << std::endl;

	for (int i = 0; i < noninternal_states_count_; i++)
	{
		if (result_occurences_[i] != 0)
			std::cout << (float)result_occurences_[i] / (float)n_trajectories << " "
					  << window_average_small_stats::non_internal_idx_to_state(noninternals_mask_, i).to_string(nodes)
					  << std::endl;
	}
}

void final_states_stats::write_csv(int, const std::vector<std::string>&, const std::string&) {}
