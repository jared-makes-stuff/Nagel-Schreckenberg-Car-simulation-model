#include "common.h"

#include <assert.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <vector>

namespace traffic_prng {
// Taken from
// https://github.com/Practical-Scientific-and-HPC-Computing/Traffic_EduHPC-23/blob/main/Code/traffic.cpp
// Global pseudo random number generator based off of C++'s minstd_rand
PRNG* engine;
std::uniform_real_distribution<float>* uniformr;  ///< uniform real distribution
std::uniform_int_distribution<int>* uniformi;  ///< uniform integer distribution

}  // namespace traffic_prng

/// @brief Function to initialize internal variables need to make the
///        pseudo-random number generation work.
///
/// @param seed Random seed for the Marsenne-Twister (in).
///
/// @return void
void random_initialize(long seed) {
  traffic_prng::engine = new PRNG(seed);
  traffic_prng::uniformr = new std::uniform_real_distribution<float>(0.0, 1.0);
  traffic_prng::uniformi = new std::uniform_int_distribution<int>();
  // note: uniformi cannot be initialized yet because we do not know the max
  // integer
}

Params parse_input_file() {
  Params p;

  std::string line;
  std::cin >> p.n >> p.L >> p.vmax >> p.p_dec >> p.p_start >> p.steps >> p.seed;

#ifdef DEBUG
  std::cerr << "n: " << p.n << '\n';
  std::cerr << "l: " << p.L << '\n';
  std::cerr << "V_max: " << p.vmax << '\n';
  std::cerr << "p_dec: " << p.p_dec << '\n';
  std::cerr << "p_start: " << p.p_start << '\n';
  std::cerr << "steps: " << p.steps << '\n';
  if (p.seed) std::cerr << "seed: " << p.seed << '\n';
#endif
  // Basic validation
  if (p.n <= 0 || p.L <= 0 || p.vmax <= 0 || p.p_dec < 0 || p.p_dec > 1.0 ||
      p.p_start < 0 || p.p_start > 1.0 || p.steps <= 0) {
    throw std::runtime_error("Invalid parameters in input file.");
  }

  return p;
}

int get_random_int(int min_val, int max_val) {
  // std::uniform_int_distribution<int> dist(min_val, max_val);
  // auto r = dist(traffic::mt);
  // return r;
  traffic_prng::uniformi->param(
      std::uniform_int_distribution<int>::param_type{min_val, max_val});
  return (*traffic_prng::uniformi)(*traffic_prng::engine);
}

bool flip_coin(double p, PRNG* engine = traffic_prng::engine) {
  auto r = (*traffic_prng::uniformr)(*engine);
  if (r < p)
    return true;
  else
    return false;
}

/// @brief Function to randomly place N agents on a grid of L points without
/// overlap.
///
/// @param  N    number of agents to place (in)
/// @param  L    number of grid points (in)
///
/// @return vector of integers with the positions of the agents, in order of
/// increasing magnitude
std::vector<int> random_placement(int N, int L) {
  std::vector<int> result(N);
  for (int i = 0; i < N; i++)
    result[i] = get_random_int(0, L - N - 1);  // allow space to shift agents
  std::sort(result.begin(), result.end());
  for (int i = 0; i < N; i++) result[i] += i;  // guaranteed to separate agents
  return result;
}

void reportResult(const std::vector<Car>& cars, int epoch) {
  (void) cars;
  (void) epoch;
#ifdef DEBUG
  std::cerr << "Epoch: " << epoch << '\n';
  for (int i = 0; i < (int) cars.size(); ++i) {
    const auto& car = cars[i];
    std::cerr << "Car " << car.id << ": Lane " << car.lane << ": Position "
              << car.position << ", Speed " << car.v << '\n';
    // if (i)
    //   fflush(stdout), assert(car.lane != cars[i - 1].lane or
    //                          car.position != cars[i - 1].position);
  }
#endif
}

void reportFinalResult(const std::vector<Car>& cars) {
  std::cout << "Correctness check:\n";
  for (int i = 0; i < std::min(10, (int)cars.size()); ++i) {
    const auto& car = cars[i];
    std::cout << "Car " << car.id << ": Lane " << car.lane << ": Position "
              << car.position << ", Speed " << car.v << '\n';
  }
}

void executeSimulation(Params, std::vector<Car>);

int main() {
  try {
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);
    std::cout.tie(NULL);
    auto params = parse_input_file();

    random_initialize(params.seed);

    std::vector<Car> cars(params.n);
    for (int i = 0; i < params.n; ++i) {
      cars[i].id = i;
      std::cin >> cars[i].lane >> cars[i].position >> cars[i].v;
      // cars[i] = {i, get_random_int(0, params.vmax), i, flip_coin(0.5)};
    }
    reportResult(cars, -1);

    executeSimulation(params, cars);
  } catch (const std::exception& e) {
    std::cout << "Fatal error: " << e.what() << "\n";
    return 1;
  }
}
