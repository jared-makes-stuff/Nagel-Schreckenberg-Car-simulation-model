#include <assert.h>
#include <omp.h>

#include <algorithm>
#include <iostream>
#include <optional>
#include <vector>

#include "common.h"

namespace traffic_prng {
extern PRNG* engine;
}

void executeSimulation(Params params, std::vector<Car> cars) {

	constexpr int MAX_VAL = (1u << (sizeof(int) * 8 - 1)) - 1;
	const size_t num_cars = cars.size();
	std::vector<char> defer_ss(num_cars, 0);
	std::vector<char> ss(num_cars, 0);
	std::vector<char> dec(num_cars, 0);
	std::vector<char> should_move(num_cars, 0);
	std::vector<int> next_v(num_cars, 0);

  for (int timestep = 0; timestep < params.steps; ++timestep) {
		// Do your calculations	
		// Extension 1: Each car calls PRNG twice for slowstart and deceleration
		for (size_t i = 0; i < num_cars; i++) {
			ss[i] = flip_coin(params.p_start, traffic_prng::engine);
			dec[i] = flip_coin(params.p_dec, traffic_prng::engine);
		}

		// Extension 2 (pt 1): Each car decides whether to change lane or not
		for (size_t i = 0; i < num_cars; i++) {
	
			// Find the distance between the nearest cars for car[i] 
			// Find the velocity of the car nearest behind on the other lane of car[i].
			const Car& self = cars[i];
			bool empty_space = true;
			int L = params.L;

			int dist_ahead_same_lane  = MAX_VAL;
			int dist_ahead_other_lane = MAX_VAL;
			int dist_behind_other_lane = MAX_VAL;
			int velocity_self = self.v;
			int velocity_behind = -1;
			
			for (size_t j = 0; j < num_cars; ++j) {
				if (j == i) {continue;}

				const Car& other = cars[j];

				int dist_ahead = (other.position - self.position + L) % L;
				if (dist_ahead == 0) {
					empty_space = false;  // There is a car at the same position on the other lane
					break;
				}
	
				bool same_lane = (other.lane == self.lane);

				if (same_lane) {
					dist_ahead_same_lane = std::min(dist_ahead_same_lane, dist_ahead);
				} 
				else {
					dist_ahead_other_lane = std::min(dist_ahead_other_lane, dist_ahead);

					int dist_behind = L - dist_ahead;
					if (dist_behind < dist_behind_other_lane) {
						dist_behind_other_lane = dist_behind;
						velocity_behind  = cars[j].v;
					}
				}
			}

			bool cond_1 = dist_ahead_other_lane > dist_ahead_same_lane;
			bool cond_2 = velocity_self >= dist_ahead_same_lane;
			bool cond_3 = empty_space;
			bool cond_4 = dist_behind_other_lane > velocity_behind;

			should_move[i] = cond_1 && cond_2 && cond_3 && cond_4;
		}

		// Extension 2 (pt 2): All cars change/not change lanes simultaneously
		for (size_t i = 0; i < num_cars; i++) {
			if (should_move[i]) {
				cars[i].lane = 1 - cars[i].lane;  // Toggle between lane 0 and 1
			}
		}

		// Extension 1: Nagel-Schreckenberg Model with Slow Start & Slow Down
		for (size_t i = 0; i < num_cars; i++) {

			bool skip_rule2 = false;
			bool skip_rule3 = false;

			// Find d (distance between car[i] and the next car in front of it)
			// Find v_ip1 (velocity of the next car infront of car[i])
			// Needed for Rule 1, Rule 2, Rule 3
			Car& self = cars[i];
			int L = params.L;

			int d  = MAX_VAL;
			next_v[i] = self.v;
			int v_ip1 = -1; 

			for (size_t j = 0; j < num_cars; ++j) {
				if (j == i) {continue;}

				const Car& other = cars[j];
				bool same_lane = (other.lane == self.lane);

				if (!same_lane) {continue;}

				int dist_ahead = (other.position - self.position + L) % L;

				if (dist_ahead < d) {
					d = dist_ahead;
					v_ip1 = cars[j].v;
				}
			}

			// Rule 1: slow start
			if (timestep != 0 && defer_ss[i]) {
				defer_ss[i] = false;
				next_v[i] = 1;
				skip_rule3 = true;
			} else if (next_v[i] == 0 && d > 1) {
				if (ss[i]) {
					defer_ss[i] = false;
					skip_rule2 = true;
				} else {
					defer_ss[i] = true;
					skip_rule2 = true;
					skip_rule3 = true;
				}
			}

			// Rule 2: avoid crashing / deceleration
			if (!skip_rule2) {

				// case 1: the car in front is moving faster than you, or the car in front is very close to you
				if (d <= next_v[i] && ((next_v[i] < v_ip1) || (next_v[i] < 2))) {
					next_v[i] = d - 1;
					skip_rule3 = true;
				} 
				
				// case 2: the car in front is moving faster than you, or the car in front is very close to you
				else if (d <= next_v[i] && (next_v[i] >= v_ip1) && next_v[i] >= 2) {
					next_v[i] = std::min(d - 1, next_v[i] - 2);
					skip_rule3 = true;
				}

				// case 3: the car ahead is moving slower than you, and you foresee a collision in 2 timesteps
				else if (next_v[i] < d && d <= 2*next_v[i] && next_v[i] >= v_ip1) {
					next_v[i] = next_v[i] - (int)((next_v[i] - v_ip1)/2);
					skip_rule3 = true;  // Edge case assumes where (v_i - v_ip2) = 1, cars[i] velocity never changes but still counts as being modified
				}
			}

			// Rule 3: acceleration
			if (!skip_rule3) {
				next_v[i] = std::min(d - 1, std::min(next_v[i] + 1, params.vmax));
			}

			// Rule 4: random deceleration
			if (next_v[i] > 0 && dec[i]) {
				next_v[i] -= 1;
			}
		}

		// Rule 5: Move the cars forwards by their respective vi units.
		for (size_t i = 0; i < num_cars; i++) {
			
			cars[i].position = (cars[i].position + next_v[i]) % params.L;
			cars[i].v = next_v[i];
		}
		// Report the result for each timestep.
		// Make sure you update your cars vector appropriately.
		reportResult(cars, timestep);
	}
	// Report the final state of the cars.
	reportFinalResult(cars);	
  }
