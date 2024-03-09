#include <curand_kernel.h>

#include <thrust/device_free.h>
#include <thrust/device_malloc.h>
#include <thrust/iterator/iterator_adaptor.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/partition.h>
#include <thrust/host_vector.h>

#include <cmath>

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

	__host__ __device__ bool operator()(T other) { return other == it; }
};


template <typename T>
struct logical_not_bitwise_and
{
	T it;

	logical_not_bitwise_and(T it) : it(it) {}

	__host__ __device__ bool operator()(T other) {
		return 0 == (other && it); 
	}
};


template <typename T>
struct bitwise_and
{
	T it;

	bitwise_and(T it) : it(it) {}

	__host__ __device__ bool operator()(T other) { 
		return other && it; 
		
	}
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

template<typename Iterator>
class jump_iterator : public thrust::iterator_adaptor<jump_iterator<Iterator>,Iterator>
{
  public:
    typedef thrust::iterator_adaptor<jump_iterator<Iterator>,Iterator> super_t;
    __host__ __device__ jump_iterator(Iterator x, int n) : super_t(x), begin(x), n(n) {}
    friend class thrust::iterator_core_access;

  private:
    unsigned int n;
    Iterator begin;
    __host__ __device__ void increment() {this->base_reference()+=n;} 
};








void simulation_runner::save_trajs_before_overwrite(int trajectories_in_batch,int new_batch_addition,thrust::device_ptr<state_word_t> d_last_states,thrust::device_ptr<float> d_last_times, thrust::device_ptr<curandState> d_rands)
{
	//saved anyway for external_inputs calc (this may change when more of that is put onto the GPU
	thrust::host_vector<state_word_t> state_buffer(new_batch_addition*state_words_);
	cudaMemcpy(state_buffer.data(), d_last_states.get()+trajectories_in_batch*state_words_,  new_batch_addition*state_words_ * sizeof(state_word_t),cudaMemcpyDeviceToHost);
	saved_states.insert(saved_states.end(), state_buffer.begin(), state_buffer.begin() + new_batch_addition*state_words_);//check size of vector

	thrust::host_vector<float> times_buffer(new_batch_addition);
	cudaMemcpy(times_buffer.data(), d_last_times.get()+trajectories_in_batch, new_batch_addition * sizeof(float),cudaMemcpyDeviceToHost);
	saved_times.insert(saved_times.end(), times_buffer.begin(), times_buffer.begin() + new_batch_addition);

	//is it neccasary to make sure rands are kept in order? they are just random seeds after all
	thrust::host_vector<curandState> rands_buffer(new_batch_addition);
	cudaMemcpy(rands_buffer.data(), d_rands.get()+trajectories_in_batch, new_batch_addition * sizeof(curandState),cudaMemcpyDeviceToHost); 
	saved_rands.insert(saved_rands.end(),rands_buffer.begin(),rands_buffer.begin()+new_batch_addition);
}

void simulation_runner::load_batch_addition_from_saved_and_new(int trajectories_in_batch,int new_batch_addition,thrust::device_ptr<state_word_t> d_last_states,thrust::device_ptr<float> d_last_times, thrust::device_ptr<curandState> d_rands)
{
	//saved anyway for external_inputs calc (this may change when more of that is put onto the GPU
	cudaMemcpy(d_last_states.get()+trajectories_in_batch*state_words_, saved_states.data(), new_batch_addition*state_words_ * sizeof(state_word_t),cudaMemcpyHostToDevice);
	saved_states.erase(saved_states.begin(), saved_states.begin() + new_batch_addition* state_words_ );
	
	cudaMemcpy(d_last_times.get()+trajectories_in_batch, saved_times.data(), new_batch_addition * sizeof(float),cudaMemcpyHostToDevice);
	saved_times.erase(saved_times.begin(), saved_times.begin() + new_batch_addition);

	//is it neccasary to make sure rands are kept in order? they are just random seeds after all
	cudaMemcpy(d_rands.get()+trajectories_in_batch, saved_rands.data() , new_batch_addition * sizeof(curandState),cudaMemcpyHostToDevice); 
	saved_rands.erase(saved_rands.begin(), saved_rands.begin() + new_batch_addition);
}




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



	thrust::device_ptr<float> external_inputs = thrust::device_malloc<float>(drv.external_inputs.size());
	std::vector<float> external_inputs_host(drv.external_inputs.size());
	std::vector<state_word_t> h_last_states(trajectory_batch_limit * state_words_);

	
	int steps;
	if (drv.constants.find("steps") != drv.constants.end())
		steps = drv.constants.at("steps");
	else
		steps = 1;


	bool death_and_div_set = true;
	auto it = std::find_if(drv.nodes.begin(), drv.nodes.end(), [this](auto&& node) { return node.name == "Death"; });
	if (it == drv.nodes.end())
	{
		death_and_div_set = false;
	}
	int i = it - drv.nodes.begin();
	int death_word_offset = i / 32;
	int death_bit = i % 32;

	it = std::find_if(drv.nodes.begin(), drv.nodes.end(), [this](auto&& node) { return node.name == "Division"; });
	if (it == drv.nodes.end())
	{
		death_and_div_set = false;
	}
	i = it - drv.nodes.begin();
	int division_word_offset = i / 32;
	int division_bit = i % 32;

	for(int step = 0; step< steps; step++)
	{	
		//may put this in the kernel
		h_last_states.resize(trajectory_batch_limit*state_words_+saved_states.size()*state_words_);
		CUDA_CHECK(cudaMemcpy(h_last_states.data(), d_last_states.get(), trajectory_batch_limit * state_words_ * sizeof(state_word_t),cudaMemcpyDeviceToHost));
	    std::copy(saved_states.begin(), saved_states.end(), h_last_states.begin()+trajectory_batch_limit * state_words_);
		for (int i = 0; i < drv.external_inputs.size();i++)
		{
			external_inputs_host[i] = drv.external_inputs.at(i).expr->evaluate(drv,h_last_states);
		}
		CUDA_CHECK(cudaMemcpy(external_inputs.get(), external_inputs_host.data(), external_inputs_host.size() * sizeof(float),cudaMemcpyHostToDevice));


		if (death_and_div_set && step > 0)		
		{
			//do not use h_last_states!
			auto h_times_rands = thrust::make_zip_iterator(saved_times.begin(), saved_rands.begin());


			auto d_times_rands = thrust::make_zip_iterator(d_last_times, d_rands);

			//handling division
			auto d_division_idx = thrust::stable_partition(d_times_rands,d_times_rands+trajectory_batch_limit,
				 jump_iterator(d_last_states+division_word_offset, state_words_),//that does the indexes
				 bitwise_and<state_word_t>(pow(2,division_bit))) - d_times_rands;


			thrust::stable_partition(d_last_states,d_last_states+trajectory_batch_limit,
				 repeat_iterator(jump_iterator(d_last_states+division_word_offset, state_words_),state_words_),//that does the indexes
				 bitwise_and<state_word_t>(pow(2,division_bit)));


			auto h_division_idx = thrust::stable_partition(thrust::host,h_times_rands,h_times_rands+static_cast<int>(saved_rands.size()),
				 jump_iterator(saved_states.begin()+division_word_offset, state_words_),//that does the indexes
				 bitwise_and<state_word_t>(pow(2,division_bit))) - h_times_rands;

			//you do need this
			thrust::stable_partition(thrust::host,saved_states.begin(),saved_states.end(),
				 repeat_iterator(jump_iterator(saved_states.begin()+division_word_offset, state_words_),state_words_),//that does the indexes
				 bitwise_and<state_word_t>(pow(2,division_bit)));



			//handling death
			//this will put the cells which are not dead at begginging and d_death_idx points to the last live cell
			//
			auto d_death_idx = thrust::stable_partition(d_times_rands,d_times_rands+trajectory_batch_limit,
				 jump_iterator(d_last_states+death_word_offset, state_words_),//that does the indexes
				 logical_not_bitwise_and<state_word_t>(pow(2,death_bit))) - d_times_rands;

			thrust::stable_partition(d_last_states,d_last_states+trajectory_batch_limit,
				 repeat_iterator(jump_iterator(d_last_states+death_word_offset, state_words_),state_words_),//that does the indexes
				 logical_not_bitwise_and<state_word_t>(pow(2,death_bit)));


			auto h_death_idx = thrust::stable_partition(thrust::host,h_times_rands,h_times_rands+static_cast<int>(saved_rands.size()),
				 jump_iterator(saved_states.begin()+death_word_offset, state_words_),//that does the indexes
				 logical_not_bitwise_and<state_word_t>(pow(2,death_bit))) - h_times_rands;

			thrust::stable_partition(thrust::host,saved_states.begin(),saved_states.end(),
				 repeat_iterator(jump_iterator(saved_states.begin()+death_word_offset, state_words_),state_words_),//that does the indexes
				 logical_not_bitwise_and<state_word_t>(pow(2,death_bit)));


		

			int device_deaths;
			if (sample_size >= trajectory_batch_limit)
				device_deaths =  (trajectory_batch_limit - d_death_idx);
			else
				device_deaths = (sample_size - d_death_idx);

			int host_deaths = (static_cast<int>(saved_rands.size()) - h_death_idx);
			int deaths_this_step = device_deaths + host_deaths;
			int divs_this_step =  d_division_idx+h_division_idx;

		//erasing dead cells
			//from host
			saved_states.erase(saved_states.begin() + h_death_idx*state_words_,saved_states.end());
			saved_times.erase(saved_times.begin()+ h_death_idx,saved_times.end());
			saved_rands.erase(saved_rands.begin()+ h_death_idx,saved_rands.end());

		//making new trajectories
			// from host vector 'saved'
	        int originalSize = saved_times.size();

	        //should new rands be made?
	        saved_rands.resize(originalSize + h_division_idx);
	        std::copy(saved_rands.begin(), saved_rands.begin() + h_division_idx, saved_rands.begin() + originalSize); // h_division_idx is valid still 

	        //should not need to copy times, new times should be 0?
	        saved_times.resize(originalSize + h_division_idx);
	        std::copy(saved_times.begin(), saved_times.begin() + h_division_idx, saved_times.begin() + originalSize);

	        saved_states.resize(originalSize*state_words_ + h_division_idx*state_words_);
	        std::copy(saved_states.begin(), saved_states.begin() + h_division_idx*state_words_, saved_states.begin() + originalSize*state_words_);
			
		//making new trajectories
	    	//from device arrays
			save_trajs_before_overwrite(0,d_division_idx,d_last_states,d_last_times, d_rands);//d_division_idx still valid
			
		//erasing dead cells on device
			//by overwrite
			int batch_free_size = trajectory_batch_limit - d_death_idx;
			int new_batch_addition = std::min(batch_free_size, static_cast<int>(saved_rands.size()));		
			load_batch_addition_from_saved_and_new(d_death_idx,new_batch_addition, d_last_states,d_last_times, d_rands);
	

			sample_size -= deaths_this_step;
        	sample_size += divs_this_step;
		}

		n_trajectories_ = sample_size;
		int trajectories_in_batch = std::min(n_trajectories_, trajectory_batch_limit);
		n_trajectories_ -= trajectories_in_batch;

		remaining_trajs = n_trajectories_;//this var doesnt matter much


		CUDA_CHECK(cudaMemset(d_last_times.get(), 0,  trajectory_batch_limit * sizeof(float)));
		CUDA_CHECK(cudaMemset(d_traj_times.get(), 0,  trajectory_batch_limit * trajectory_len_limit * sizeof(float)));

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
						save_trajs_before_overwrite(trajectories_in_batch,new_batch_addition, d_last_states,d_last_times, d_rands);
						if (step == 0)
						{
							initialize_initial_state.run(
								dim3(DIV_UP(new_batch_addition, 256)), dim3(256), new_batch_addition, state_size_,
								d_initial_probs.get(), d_last_states.get() + trajectories_in_batch * state_words_,
								d_last_times.get() + trajectories_in_batch, d_rands.get() + trajectories_in_batch);
						}
						else
						{
							load_batch_addition_from_saved_and_new(trajectories_in_batch,new_batch_addition, d_last_states,d_last_times, d_rands);

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