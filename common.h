#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

class Params {
 public:
  int n;                           // number of cars
  int L;                           // road length (cells)
  int vmax;                        // maximum velocity (cells/step)
  double p_dec;                          // random deceleration probability
  double p_start;                        // slow-start probability
  int steps;                       // number of time steps to simulate
  std::optional<std::string> init_file;  // optional initial state file
  long seed;                             // RNG seed
};

class Car {
 public:
  int id;
  int v;
  int position;
  int lane;
};

class PRNG : public std::minstd_rand {
 public:
  PRNG(result_type x = default_seed) : std::minstd_rand(x) {}
  void discard(result_type z) {
    // Faster forward routine for this multiplicative linear
    // congruent generator, O(log z) instead of O(z)
    // compute az = multiplier^z mod modulus:
    result_type b = multiplier, az = 1, x;
    while (z > 0) {
      if (z % 2) az = (az * b) % modulus;
      b = (b * b) % modulus;
      z >>= 1;
    }
    // apply to current state:
    std::stringstream s;
    s << (*this);              // get state as a string
    s >> x;                    // convert back to unsigned long
    seed((x * az) % modulus);  // set forwarded state
  }
};

int get_random_int(int, int);
bool flip_coin(double, PRNG* engine);
void reportResult(const std::vector<Car>&, int);
void reportFinalResult(const std::vector<Car>&);

void executeSimulation(Params, std::vector<Car>);
