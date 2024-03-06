#include <curand_kernel.h>

using uint8_t = unsigned char;
using uint32_t = unsigned int;

#include "../state_word.h"
#include "../trajectory_status.h"

__device__ int select_flip_bit(int state_size, const float* __restrict__ transition_rates, float total_rate,
							   curandState* __restrict__ rand)
{
	float r = curand_uniform(rand) * total_rate;
	float sum = 0;
	int idx = 0;
	for (int i = 0; i < state_size; i++)
	{
		sum += transition_rates[i];
		idx += (sum < r) ? 1 : 0;
	}
	return idx;
}

extern "C" __global__ void initialize_random(int trajectories_count, unsigned long long seed,
											 curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	// initialize random number generator
	curand_init(seed, id, 0, rands + id);
}

extern "C" __global__ void initialize_initial_state(int trajectories_count, int state_size,
													const float* __restrict__ initial_probs,
													state_word_t* __restrict__ states, float* __restrict__ times,
													curandState* __restrict__ rands)
{
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	constexpr int word_size = sizeof(state_word_t) * 8;
	int state_words = (state_size + word_size - 1) / word_size;

	// initialize state
	{
		state_word_t s = 0;

		// randomly set free vars
		for (int i = 0; i < state_size; i++)
		{
			if (curand_uniform(rands + id) <= initial_probs[i])
				s |= 1 << (i % word_size);
			else
				s &= ~(1 << (i % word_size));

			if (i % word_size == word_size - 1)
			{
				states[id * state_words + i / word_size] = s;
				s = 0;
			}
		}

		if (state_size % word_size != 0)
			states[id * state_words + state_words - 1] = s;
	}

	// set time to zero
	times[id] = 0.f;
}

extern __device__ float compute_transition_rates(float* __restrict__ transition_rates,
												 const state_word_t* __restrict__ state,float* external_inputs);
extern __device__ float compute_transition_entropy(const float* __restrict__ transition_rates);

__device__ void simulate_inner(int trajectories_count, int state_size, int trajectory_limit, float time_tick,
							   float max_time, bool discrete_time, state_word_t* __restrict__ last_states,
							   float* __restrict__ last_times, void* __restrict__ rands_v,
							   state_word_t* __restrict__ trajectory_states, float* __restrict__ trajectory_times,
							   float* __restrict__ trajectory_transition_entropies,
							   trajectory_status* __restrict__ trajectory_statuses,
							   float* __restrict__ transition_rates, state_word_t* __restrict__ state,float* external_inputs)
{
	curandState* __restrict__ rands = reinterpret_cast<curandState*>(rands_v);
	auto id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id >= trajectories_count)
		return;

	constexpr int word_size = sizeof(state_word_t) * 8;
	int state_words = (state_size + word_size - 1) / word_size;

	for (int i = 0; i < state_words; i++)
		state[i] = last_states[id * state_words + i];
	curandState rand = rands[id];
	float time = last_times[id];
	int step = 0;
	trajectory_states = trajectory_states + id * state_words * trajectory_limit;
	trajectory_times = trajectory_times + id * trajectory_limit;
	trajectory_transition_entropies = trajectory_transition_entropies + id * trajectory_limit;
	trajectory_status status = trajectory_status::CONTINUE;

	// as the first time set the last from the prev run
	trajectory_times[step++] = time;

	while (true)
	{
		// get transition rates for current state
		float total_rate = compute_transition_rates(transition_rates, state,external_inputs);

		float transition_entropy = 0.f;

		// if total rate is zero, no transition is possible
		if (total_rate == 0.f)
		{
			status = trajectory_status::FIXED_POINT;
			time = max_time;
		}
		else
		{
			if (discrete_time)
			{
				time += time_tick;
				// due to a common rounding error for discrete simulations,
				// we set time to max_time explicitely if it is close enough to max_time
				time = max_time - time_tick < time ? max_time : time;
			}
			else
				time += -logf(curand_uniform(&rand)) / total_rate;

			time = fminf(time, max_time);

			// if total rate is nonzero, we compute the transition entropy
			transition_entropy = compute_transition_entropy(transition_rates);
		}

		for (int i = 0; i < state_words; i++)
			trajectory_states[step * state_words + i] = state[i];
		trajectory_times[step] = time;
		trajectory_transition_entropies[step] = transition_entropy;
		step++;

		if (time >= max_time || step >= trajectory_limit)
			break;

		int flip_bit = select_flip_bit(state_size, transition_rates, total_rate, &rand);
		state[flip_bit / word_size] ^= 1 << (flip_bit % word_size);
	}

	// save thread variables
	{
		for (int i = 0; i < state_words; i++)
			last_states[id * state_words + i] = state[i];
		rands[id] = rand;
		last_times[id] = time;
	}

	if (status != trajectory_status::FIXED_POINT)
	{
		status = (time >= max_time) ? trajectory_status::FINISHED : trajectory_status::CONTINUE;
	}

	trajectory_statuses[id] = status;
}
