#include "window_average_small.h"

#include <cmath>
#include <fstream>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>

#include "../timer.h"

window_average_small_stats::window_average_small_stats(float window_size, float max_time, bool discrete_time,
													   state_t noninternals_mask, size_t non_internals,
													   size_t max_traj_len, size_t max_n_trajectories,
													   kernel_wrapper& window_average_small)
	: window_size_(window_size),
	  max_time_(max_time),
	  discrete_time_(discrete_time),
	  noninternal_states_count_(1 << non_internals),
	  noninternals_mask_(std::move(noninternals_mask)),
	  max_traj_len_(max_traj_len),
	  max_n_trajectories_(max_n_trajectories),
	  window_average_small_(window_average_small)
{
	timer_stats stats("window_average_small> initialize");

	size_t windows_count = std::ceil(max_time / window_size);

	window_tr_entropies_ = thrust::device_malloc<float>(windows_count);
	result_tr_entropies_.resize(windows_count);
	CUDA_CHECK(cudaMemset(window_tr_entropies_.get(), 0, windows_count * sizeof(float)));

	if (discrete_time)
	{
		window_probs_discrete_ = thrust::device_malloc<int>(windows_count * noninternal_states_count_);
		result_probs_discrete_.resize(windows_count * noninternal_states_count_);
		CUDA_CHECK(
			cudaMemset(window_probs_discrete_.get(), 0, windows_count * noninternal_states_count_ * sizeof(int)));
	}
	else
	{
		window_probs_ = thrust::device_malloc<float>(windows_count * noninternal_states_count_);
		result_probs_.resize(windows_count * noninternal_states_count_);
		CUDA_CHECK(cudaMemset(window_probs_.get(), 0, windows_count * noninternal_states_count_ * sizeof(float)));
	}
}

window_average_small_stats::~window_average_small_stats()
{
	timer_stats stats("window_average_small> free");

	thrust::device_free(window_probs_);
	thrust::device_free(window_probs_discrete_);
	thrust::device_free(window_tr_entropies_);
}

void window_average_small_stats::process_batch(thrust::device_ptr<state_word_t> traj_states,
											   thrust::device_ptr<float> traj_times,
											   thrust::device_ptr<float> traj_tr_entropies,
											   thrust::device_ptr<state_word_t>, thrust::device_ptr<trajectory_status>,
											   int n_trajectories)
{
	process_batch_internal(traj_states, traj_times, traj_tr_entropies, n_trajectories);
}

void window_average_small_stats::process_batch_internal(thrust::device_ptr<state_word_t> traj_states,
														thrust::device_ptr<float> traj_times,
														thrust::device_ptr<float> traj_tr_entropies, int n_trajectories)
{
	timer_stats stats("window_average_small> process_batch");

	int windows_count = std::ceil(max_time_ / window_size_);
	int shared_mem_size = sizeof(float) * windows_count;

	bool extra_shared_mem = false;

	assert(sizeof(float) == sizeof(int));

	if (shared_mem_size + sizeof(float) * noninternal_states_count_ * windows_count < 10 * 1024)
	{
		shared_mem_size += sizeof(float) * noninternal_states_count_ * windows_count;
		extra_shared_mem = true;
	}
	//sort to put the 'empty' trajes to the end, then count the index of the first traj, then num threads in kernel to be number of non-free trajes.
	//traj stuatus being used to denote 'not a trajectory' 
	window_average_small_.run_shared(
		dim3(DIV_UP(n_trajectories * (max_traj_len_ - 1), 256)), dim3(256), shared_mem_size, max_traj_len_,
		n_trajectories, (int)noninternals_mask_.words_n(), noninternal_states_count_, window_size_, windows_count,
		extra_shared_mem, traj_states.get(), traj_times.get(), traj_tr_entropies.get(),
		discrete_time_ ? (void*)window_probs_discrete_.get() : (void*)window_probs_.get(), window_tr_entropies_.get());
}

void window_average_small_stats::finalize()
{
	timer_stats stats("window_average_small> finalize");

	size_t windows_count = std::ceil(max_time_ / window_size_);
	timer t;

	// copy result data into host
	if (discrete_time_)
		CUDA_CHECK(cudaMemcpy(result_probs_discrete_.data(), window_probs_discrete_.get(),
							  windows_count * noninternal_states_count_ * sizeof(int), cudaMemcpyDeviceToHost));
	else
		CUDA_CHECK(cudaMemcpy(result_probs_.data(), window_probs_.get(),
							  windows_count * noninternal_states_count_ * sizeof(float), cudaMemcpyDeviceToHost));
	CUDA_CHECK(cudaMemcpy(result_tr_entropies_.data(), window_tr_entropies_.get(), windows_count * sizeof(float),
						  cudaMemcpyDeviceToHost));
}

state_t window_average_small_stats::non_internal_idx_to_state(const state_t& noninternals_mask, int idx)
{
	state_t ret(noninternals_mask.state_size);
	size_t idx_i = 0;
	for (size_t i = 0; i < noninternals_mask.state_size; i++)
	{
		if (noninternals_mask.is_set(i))
		{
			if ((idx & (1 << idx_i)) != 0)
				ret.set(i);
			idx_i++;
		}
	}

	return ret;
}

float window_average_small_stats::get_single_result_prob(int n_trajectories, size_t idx)
{
	if (discrete_time_)
	{
		auto occurences = result_probs_discrete_[idx];

		return (float)occurences / (float)n_trajectories;
	}
	else
	{
		auto cumul_slices = result_probs_[idx];

		return cumul_slices / (n_trajectories * window_size_);
	}
}

void window_average_small_stats::visualize(int n_trajectories, const std::vector<std::string>& nodes)
{
	timer_stats stats("window_average_small> visualize");

	size_t windows_count = std::ceil(max_time_ / window_size_);

	for (size_t i = 0; i < windows_count; ++i)
	{
		float entropy = 0.f;
		float wnd_tr_entropy = result_tr_entropies_[i] / n_trajectories;
		wnd_tr_entropy /= discrete_time_ ? 1 : window_size_;

		for (uint32_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
		{
			auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

			if (prob == 0.f)
				continue;

			entropy += -std::log2(prob) * prob;
		}

		std::cout << "window (" << i * window_size_ << ", " << (i + 1) * window_size_ << "]" << std::endl;
		std::cout << "entropy: " << entropy << std::endl;
		std::cout << "transition entropy: " << wnd_tr_entropy << std::endl;

		for (uint32_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
		{
			auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

			if (prob == 0.f)
				continue;

			std::cout << prob << " " << non_internal_idx_to_state(noninternals_mask_, s_idx).to_string(nodes)
					  << std::endl;
		}
	}
}

void window_average_small_stats::write_csv(int n_trajectories, const std::vector<std::string>& nodes,
										   const std::string& prefix)
{
	timer_stats stats("window_average_small> write_csv");

	size_t windows_count = std::ceil(max_time_ / window_size_);
	std::ofstream ofs;

	ofs.open(prefix + "_probtraj.csv");
	if (ofs)
	{
		// Computing max states for header
		int max_states = 0;
		for (size_t i = 0; i < windows_count; ++i)
		{
			int num_states = 0;
			for (uint32_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;

				num_states += 1;
			}

			max_states = std::max(max_states, num_states);
		}

		// Writing header
		ofs << "Time\tTH\tErrorTH\tH\tHD=0";
		for (int i = 0; i < max_states; i++)
		{
			ofs << "\tState\tProba\tErrorProba";
		}
		ofs << std::endl;

		// Writing trajectories
		for (size_t i = 0; i < windows_count; ++i)
		{
			float entropy = 0.f;
			float wnd_tr_entropy = result_tr_entropies_[i] / n_trajectories;
			wnd_tr_entropy /= discrete_time_ ? 1 : window_size_;

			for (uint32_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;

				entropy += -std::log2(prob) * prob;
			}
			ofs << i * window_size_ << "\t";
			ofs << wnd_tr_entropy << "\t" << 0.f << "\t" << entropy << "\t" << 0.f;

			for (uint32_t s_idx = 0; s_idx < noninternal_states_count_; s_idx++)
			{
				auto prob = get_single_result_prob(n_trajectories, i * noninternal_states_count_ + s_idx);

				if (prob == 0.f)
					continue;

				ofs << "\t" << non_internal_idx_to_state(noninternals_mask_, s_idx).to_string(nodes) << "\t" << prob
					<< "\t" << 0.f;
			}
			ofs << std::endl;
		}
	}
}
