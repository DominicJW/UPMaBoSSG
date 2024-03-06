#include <curand_kernel.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>

#include "simulation_runner.h"
#include "state_word.h"
#include "timer.h"
#include "utils.h"
#include "./parser/driver.h"

template <typename T>
struct eq_ftor
{
	T it;

	eq_ftor(T it) : it(it) {}

	__device__ bool operator()(T other) { return other == it; }
};

template <typename Iterator>
class repeat_iterator : public thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator>
{
public:
	typedef thrust::iterator_adaptor<repeat_iterator<Iterator>, Iterator> super_t;
	__host__ __device__ repeat_iterator(const Iterator& x, int n) : super_t(x), begin(x), n(n) {}
	friend class thrust::iterator_core_access;

private:
	unsigned int n;
	Iterator begin;

	__host__ __device__ typename super_t::reference dereference() const
	{
		return *(begin + (this->base() - begin) / n);
	}
};

simulation_runner::simulation_runner(int n_trajectories, int state_size, unsigned long long seed,
									 std::vector<float> inital_probs,driver& drv)
	: n_trajectories_(n_trajectories),
	  state_size_(state_size),
	  state_words_(DIV_UP(state_size, 32)),
	  seed_(seed),
	  inital_probs_(std::move(inital_probs)),
	  drv(drv)
{
	trajectory_batch_limit = std::min(1'000'000, n_trajectories);
	trajectory_len_limit = 100; // TODO compute limit according to the available mem
}

void simulation_runner::run_simulation(stats_composite& stats_runner, kernel_wrapper& initialize_random,
									   kernel_wrapper& initialize_initial_state, kernel_wrapper& simulate)
{
	int remaining_trajs = n_trajectories_;
	int sample_size = n_trajectories_;

	thrust::device_ptr<state_word_t> d_last_states;
	thrust::device_ptr<float> d_last_times;
	thrust::device_ptr<curandState> d_rands;
	thrust::device_ptr<float> d_initial_probs;

	thrust::device_ptr<state_word_t> d_traj_states;
	thrust::device_ptr<float> d_traj_times;
	thrust::device_ptr<float> d_traj_tr_entropies;
	thrust::device_ptr<trajectory_status> d_traj_statuses;

	{
		timer_stats stats("simulation_runner> allocate");

		d_last_states = thrust::device_malloc<state_word_t>(trajectory_batch_limit * state_words_);
		d_last_times = thrust::device_malloc<float>(trajectory_batch_limit);
		d_rands = thrust::device_malloc<curandState>(trajectory_batch_limit);
		d_initial_probs = thrust::device_malloc<float>(inital_probs_.size());

		d_traj_states =
			thrust::device_malloc<state_word_t>(trajectory_batch_limit * trajectory_len_limit * state_words_);
		d_traj_times = thrust::device_malloc<float>(trajectory_batch_limit * trajectory_len_limit);
		d_traj_tr_entropies = thrust::device_malloc<float>(trajectory_batch_limit * trajectory_len_limit);
		d_traj_statuses = thrust::device_malloc<trajectory_status>(trajectory_batch_limit);
	}

	// initialize states
	{
		timer_stats stats("simulation_runner> initialize");

		CUDA_CHECK(cudaMemcpy(d_initial_probs.get(), inital_probs_.data(), inital_probs_.size() * sizeof(float),
							  cudaMemcpyHostToDevice));

		initialize_random.run(dim3(DIV_UP(trajectory_batch_limit, 256)), dim3(256), trajectory_batch_limit, seed_,
							  d_rands.get());

		initialize_initial_state.run(dim3(DIV_UP(trajectory_batch_limit, 256)), dim3(256), trajectory_batch_limit,
									 state_size_, d_initial_probs.get(), d_last_states.get(), d_last_times.get(),
									 d_rands.get());

		CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, trajectory_batch_limit * trajectory_len_limit * sizeof(float)));
	}
	int trajectories_in_batch = std::min(n_trajectories_, trajectory_batch_limit);
	n_trajectories_ -= trajectories_in_batch;



	thrust::device_ptr<float> external_inputs = thrust::device_malloc<float>(drv.external_inputs.size());
	std::vector<float> external_inputs_host(drv.external_inputs.size());	 
	std::vector<state_word_t> h_last_states(trajectory_batch_limit * state_words_);


	bool keyExists = (drv.constants.find("steps") != drv.constants.end());
	int steps;
	if (keyExists)
		steps = drv.constants.at("steps");
	else
		steps = 1;
	for(int step = 0; step< steps; step++)
	{
		n_trajectories_ = sample_size;
		remaining_trajs = n_trajectories_;;
		trajectories_in_batch = std::min(n_trajectories_, trajectory_batch_limit);

		CUDA_CHECK(cudaMemset(d_traj_statuses.get(), static_cast<int>(trajectory_status::CONTINUE),  trajectory_batch_limit * sizeof(trajectory_status)));
		CUDA_CHECK(cudaMemset(d_last_times.get(), 0,  trajectory_batch_limit * sizeof(float)));
		CUDA_CHECK(cudaMemset(d_traj_times.get(), 0,  trajectory_batch_limit * trajectory_len_limit * sizeof(float)));


		CUDA_CHECK(cudaMemcpy(h_last_states.data(), d_last_states.get(), h_last_states.size() * sizeof(state_word_t),cudaMemcpyDeviceToHost));

		for (int i = 0; i < drv.external_inputs.size();i++)
		{

			external_inputs_host[i] = drv.external_inputs.at(i).expr->evaluate(drv,h_last_states);

		}
		CUDA_CHECK(cudaMemcpy(external_inputs.get(), external_inputs_host.data(), external_inputs_host.size() * sizeof(float),cudaMemcpyHostToDevice));
		//cell death and division happens afrer this calculation


		while (trajectories_in_batch)
		{
			{
				timer_stats stats("simulation_runner> simulate");

				// run single simulation
				simulate.run(dim3(DIV_UP(trajectories_in_batch, 256)), dim3(256), trajectories_in_batch,
							 trajectory_len_limit, d_last_states.get(), d_last_times.get(), d_rands.get(),
							 d_traj_states.get(), d_traj_times.get(), d_traj_tr_entropies.get(), d_traj_statuses.get(),external_inputs.get());

			}

			{

				timer_stats stats("simulation_runner> stats");

				// compute statistics over the simulated trajs
				// restrict to final step, until new upmaboss stats class
				if (step == steps -1)
				{
				stats_runner.process_batch(d_traj_states, d_traj_times, d_traj_tr_entropies, d_last_states, d_traj_statuses,
										   trajectories_in_batch);
				}
			}

			// prepare for the next iteration
			{
				timer_stats stats("simulation_runner> prepare_next_iter");
				{
					thrust::stable_partition(d_last_states, d_last_states + trajectories_in_batch * state_words_,
											 repeat_iterator(d_traj_statuses, state_words_),
											 eq_ftor<trajectory_status>(trajectory_status::CONTINUE));


					auto thread_state_begin = thrust::make_zip_iterator(d_last_times, d_rands);
					auto remaining_trajectories_in_batch =
						thrust::stable_partition(thread_state_begin, thread_state_begin + trajectories_in_batch, d_traj_statuses,
										  eq_ftor<trajectory_status>(trajectory_status::CONTINUE))
						- thread_state_begin;

					remaining_trajs -= trajectories_in_batch - remaining_trajectories_in_batch;
					trajectories_in_batch = remaining_trajectories_in_batch;
				}


				// add new work to the batch
				{
					int batch_free_size = trajectory_batch_limit - trajectories_in_batch;
					int new_batch_addition = std::min(batch_free_size, n_trajectories_);

					if (new_batch_addition)
					{
						//why if step==0?
						if (step == 0)
						{
							initialize_initial_state.run(
								dim3(DIV_UP(new_batch_addition, 256)), dim3(256), new_batch_addition, state_size_,
								d_initial_probs.get(), d_last_states.get() + trajectories_in_batch * state_words_,
								d_last_times.get() + trajectories_in_batch, d_rands.get() + trajectories_in_batch);
						}
						trajectories_in_batch += new_batch_addition;
						n_trajectories_ -= new_batch_addition;
					}
				}

				// set all batch traj times to 0
				CUDA_CHECK(cudaMemset(d_traj_times.get(), 0, trajectories_in_batch * trajectory_len_limit * sizeof(float)));
			}

			if (timer_stats::enable_diags())
			{
				std::cerr << "simulation_runner> remaining trajs: " << remaining_trajs << std::endl;
			}
		}
	}

	timer_stats stats("simulation_runner> deallocate");

	thrust::device_free(d_last_states);
	thrust::device_free(d_last_times);
	thrust::device_free(d_rands);
	thrust::device_free(d_initial_probs);
	thrust::device_free(d_traj_states);
	thrust::device_free(d_traj_times);
	thrust::device_free(d_traj_tr_entropies);
	thrust::device_free(d_traj_statuses);
	thrust::device_free(external_inputs);
}
