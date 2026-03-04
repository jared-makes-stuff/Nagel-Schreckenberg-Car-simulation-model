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

// Small constants used across lookup and scheduling paths
const int kInfDistance = 2147483647; // Use this when no car is found
const int kDenseLookupRoadLenLimit = 2000000; // Use dense lookup only on smaller roads
const size_t kDenseRebuildDivisor = 128; // Rebuild threshold is car count divided by this
const size_t kMoverListDivisor = 8; // Use move-list when lane changes are few
const int kCompactVel8Max = 255; // Largest value that fits in uint8_t
const int kCompactVel16Max = 65535; // Largest value that fits in uint16_t

// Distance helpers for circular road positions
int forward_cyclic_distance(int from, int to, int L) {
    int d = to - from;
    if (d <= 0) {
        d += L;
    }
    return d;
}

int backward_cyclic_distance(int from, int to, int L) {
    int d = from - to;
    if (d <= 0) {
        d += L;
    }
    return d;
}

// Build dense next and previous index tables for one lane
void build_next_prev_tables(const unsigned char* gen_arr, int L, unsigned char generation, int* next_idx, int* prev_idx) {
    int next_hit = -1;
    for (int i = 2 * L - 1; i >= 0; --i) {
        const int idx = (i < L) ? i : (i - L);
        if (gen_arr[idx] == generation) {
            next_hit = idx;
        }
        if (i < L) {
            next_idx[idx] = next_hit;
        }
    }

    int prev_hit = -1;
    for (int i = 0; i < 2 * L; ++i) {
        const int idx = (i < L) ? i : (i - L);
        if (gen_arr[idx] == generation) {
            prev_hit = idx;
        }
        if (i >= L) {
            prev_idx[idx] = prev_hit;
        }
    }
}

int ahead_hit_from_table(int pos, int L, const int* next_idx, int search_limit) {
    const int query_pos = (pos + 1 == L) ? 0 : (pos + 1);
    const int hit = next_idx[query_pos];
    if (hit < 0) {
        return -1;
    }
    const int d = forward_cyclic_distance(pos, hit, L);
    return (d <= search_limit) ? hit : -1;
}

int behind_hit_from_table(int pos, int L, const int* prev_idx, int search_limit) {
    const int query_pos = (pos == 0) ? (L - 1) : (pos - 1);
    const int hit = prev_idx[query_pos];
    if (hit < 0) {
        return -1;
    }
    const int d = backward_cyclic_distance(pos, hit, L);
    return (d <= search_limit) ? hit : -1;
}

// Scan helpers for sparse lookup mode
int scan_first_match(const unsigned char* gen_arr, int start_pos, int end_pos, unsigned char target_gen) {
    for (int p = start_pos; p < end_pos; ++p) {
        if (gen_arr[p] == target_gen) {
            return p - start_pos;
        }
    }
    return -1;
}

int scan_last_match(const unsigned char* gen_arr, int start_pos, int end_pos, unsigned char target_gen) {
    if (start_pos < end_pos) {
        return -1;
    }

    for (int p = start_pos; p >= end_pos; --p) {
        if (gen_arr[p] == target_gen) {
            return p;
        }
    }
    return -1;
}

int find_ahead_distance(int pos, int L, const unsigned char* gen_arr, int search_limit, unsigned char generation) {
    const int limit1 = std::min(search_limit, L - 1 - pos);
    int found_offset = scan_first_match(gen_arr, pos + 1, pos + 1 + limit1, generation);

    if (found_offset != -1) {
        return found_offset + 1;
    }

    if (limit1 < search_limit) {
        const int limit2 = search_limit - limit1;
        found_offset = scan_first_match(gen_arr, 0, limit2, generation);
        if (found_offset != -1) {
            return limit1 + 1 + found_offset;
        }
    }

    return kInfDistance;
}

template <typename VelType>
void find_behind(int pos, int L, const unsigned char* gen_arr, const VelType* lane_vel_arr, int* out_d, int* out_v, int search_limit, unsigned char generation) {
    *out_d = kInfDistance;
    *out_v = -1;

    const int limit1 = std::min(search_limit, pos);

    if (limit1 > 0) {
        const int hit = scan_last_match(gen_arr, pos - 1, pos - limit1, generation);
        if (hit != -1) {
            *out_d = pos - hit;
            *out_v = lane_vel_arr[hit];
            return;
        }
    }

    if (limit1 < search_limit) {
        const int limit2 = search_limit - limit1;
        const int hit = scan_last_match(gen_arr, L - 1, L - limit2, generation);
        if (hit != -1) {
            *out_d = pos + (L - hit);
            *out_v = lane_vel_arr[hit];
            return;
        }
    }
}

// Read or write lane speeds using the selected compact type
int read_grid_vel(bool use_u8, bool use_u16, const uint8_t* const* grid_vels_u8, const uint16_t* const* grid_vels_u16, const int* const* grid_vels_i32, int lane, int pos) {
    if (use_u8) {
        return grid_vels_u8[lane][pos];
    }
    if (use_u16) {
        return grid_vels_u16[lane][pos];
    }
    return grid_vels_i32[lane][pos];
}

void write_grid_vel(bool use_u8, bool use_u16, uint8_t* const* grid_vels_u8, uint16_t* const* grid_vels_u16, int* const* grid_vels_i32, int lane, int pos, int vel) {
    if (use_u8) {
        grid_vels_u8[lane][pos] = vel;
    } else if (use_u16) {
        grid_vels_u16[lane][pos] = vel;
    } else {
        grid_vels_i32[lane][pos] = vel;
    }
}

void rebuild_dense_tables_parallel(int road_len, unsigned char generation, unsigned char* const* grid_gens, int* const* next_idxs, int* const* prev_idxs) {
    #pragma omp sections
    {
        #pragma omp section
        build_next_prev_tables(grid_gens[0], road_len, generation, next_idxs[0], prev_idxs[0]);
        #pragma omp section
        build_next_prev_tables(grid_gens[1], road_len, generation, next_idxs[1], prev_idxs[1]);
    }
}

// Shared data needed by lane change checks
struct LaneDecisionContext {
    const unsigned char* lane_ptr;
    const int* pos_ptr;
    const int* vel_ptr;
    int road_len;
    int search_limit;
    int behind_search_limit;
    unsigned char* const* grid_gens;
    bool use_u8_grid_vels;
    bool use_u16_grid_vels;
    uint8_t* const* grid_vels_u8;
    uint16_t* const* grid_vels_u16;
    int* const* grid_vels_i32;
    bool use_dense_lookup;
    int* const* next_idxs;
    int* const* prev_idxs;
};

char decide_lane_change_for_car(size_t i, const LaneDecisionContext& ctx, unsigned char generation) {
    const int self_lane = ctx.lane_ptr[i];
    const int other_lane = 1 - self_lane;
    const int pos_i = ctx.pos_ptr[i];
    const int vel_i = ctx.vel_ptr[i];

    if (vel_i == 0) {
        return 0;
    }

    const unsigned char* self_gen = ctx.grid_gens[self_lane];
    const unsigned char* other_gen = ctx.grid_gens[other_lane];

    // Stop early if target lane has a car at the same position
    if (other_gen[pos_i] == generation) {
        return 0;
    }

    const int ahead_same_limit = std::min(ctx.search_limit, vel_i);
    int d_ahead_same = kInfDistance;
    if (ctx.use_dense_lookup) {
        const int hit_same = ahead_hit_from_table(pos_i, ctx.road_len, ctx.next_idxs[self_lane], ahead_same_limit);
        if (hit_same >= 0) {
            d_ahead_same = forward_cyclic_distance(pos_i, hit_same, ctx.road_len);
        }
    } else {
        d_ahead_same = find_ahead_distance(pos_i, ctx.road_len, self_gen, ahead_same_limit, generation);
    }
    if (vel_i < d_ahead_same) {
        return 0;
    }

    int d_ahead_other = kInfDistance;
    if (ctx.use_dense_lookup) {
        const int hit_other = ahead_hit_from_table(pos_i, ctx.road_len, ctx.next_idxs[other_lane], d_ahead_same);
        if (hit_other >= 0) {
            d_ahead_other = forward_cyclic_distance(pos_i, hit_other, ctx.road_len);
        }
    } else {
        d_ahead_other = find_ahead_distance(pos_i, ctx.road_len, other_gen, d_ahead_same, generation);
    }
    if (d_ahead_other <= d_ahead_same) {
        return 0;
    }

    int d_behind_other = kInfDistance;
    int v_behind = -1;
    if (ctx.use_dense_lookup) {
        const int hit_behind = behind_hit_from_table(pos_i, ctx.road_len, ctx.prev_idxs[other_lane], ctx.behind_search_limit);
        if (hit_behind >= 0) {
            d_behind_other = backward_cyclic_distance(pos_i, hit_behind, ctx.road_len);
            v_behind = read_grid_vel(ctx.use_u8_grid_vels, ctx.use_u16_grid_vels, ctx.grid_vels_u8, ctx.grid_vels_u16, ctx.grid_vels_i32, other_lane, hit_behind);
        }
    } else {
        if (ctx.use_u8_grid_vels) {
            find_behind(pos_i, ctx.road_len, other_gen, ctx.grid_vels_u8[other_lane], &d_behind_other, &v_behind, ctx.behind_search_limit, generation);
        } else if (ctx.use_u16_grid_vels) {
            find_behind(pos_i, ctx.road_len, other_gen, ctx.grid_vels_u16[other_lane], &d_behind_other, &v_behind, ctx.behind_search_limit, generation);
        } else {
            find_behind(pos_i, ctx.road_len, other_gen, ctx.grid_vels_i32[other_lane], &d_behind_other, &v_behind, ctx.behind_search_limit, generation);
        }
    }

    return d_behind_other > v_behind;
}
void executeSimulation(Params params, std::vector<Car> cars) {

    // Cache key params and precompute thresholds
    const size_t car_count = cars.size();
    const int road_len = params.L;
    const size_t road_len_sz = road_len;
    const int vmax = params.vmax;
    const int search_limit = std::min(2 * vmax + 5, road_len - 1); // Max distance for local scans
    const bool generation_may_wrap = params.steps >= 255;
    const int worker_threads = std::max(1, omp_get_max_threads());
    const size_t worker_threads_sz = worker_threads;
    const bool use_dense_lookup = (road_len <= kDenseLookupRoadLenLimit) && (car_count * 2 >= road_len_sz); // Dense mode toggle
    const size_t dense_rebuild_threshold = std::max<size_t>(1, car_count / kDenseRebuildDivisor); // Min lane changes before dense rebuild
    const bool use_guided_lane_decision = (car_count >= worker_threads_sz * 32) && (car_count * 2 >= road_len_sz); // Guided schedule toggle

    // Store car fields in separate arrays for faster access
    std::vector<unsigned char> lane(car_count);
    std::vector<int> pos(car_count);
    std::vector<int> vel(car_count);
    
    // Keep lane grids contiguous and use the smallest velocity type that fits
    std::vector<unsigned char> grid_gen_0_vec(road_len, 0);
    std::vector<unsigned char> grid_gen_1_vec(road_len, 0);
    const bool use_u8_grid_vels = vmax <= kCompactVel8Max;
    const bool use_u16_grid_vels = !use_u8_grid_vels && (vmax <= kCompactVel16Max);
    std::vector<uint8_t> grid_vel_0_u8_vec(use_u8_grid_vels ? road_len : 0);
    std::vector<uint8_t> grid_vel_1_u8_vec(use_u8_grid_vels ? road_len : 0);
    std::vector<uint16_t> grid_vel_0_u16_vec(use_u16_grid_vels ? road_len : 0);
    std::vector<uint16_t> grid_vel_1_u16_vec(use_u16_grid_vels ? road_len : 0);
    std::vector<int> grid_vel_0_i32_vec((use_u8_grid_vels || use_u16_grid_vels) ? 0 : road_len);
    std::vector<int> grid_vel_1_i32_vec((use_u8_grid_vels || use_u16_grid_vels) ? 0 : road_len);
    const int dense_size = use_dense_lookup ? road_len : 0;
    std::vector<int> next_idx_0_vec(dense_size);
    std::vector<int> next_idx_1_vec(dense_size);
    std::vector<int> prev_idx_0_vec(dense_size);
    std::vector<int> prev_idx_1_vec(dense_size);
    
    unsigned char* grid_gens[2] = {grid_gen_0_vec.data(), grid_gen_1_vec.data()};
    uint8_t* grid_vels_u8[2] = {use_u8_grid_vels ? grid_vel_0_u8_vec.data() : nullptr, use_u8_grid_vels ? grid_vel_1_u8_vec.data() : nullptr};
    uint16_t* grid_vels_u16[2] = {use_u16_grid_vels ? grid_vel_0_u16_vec.data() : nullptr, use_u16_grid_vels ? grid_vel_1_u16_vec.data() : nullptr};
    int* grid_vels_i32[2] = {(use_u8_grid_vels || use_u16_grid_vels) ? nullptr : grid_vel_0_i32_vec.data(), (use_u8_grid_vels || use_u16_grid_vels) ? nullptr : grid_vel_1_i32_vec.data()};
    int* next_idxs[2] = {use_dense_lookup ? next_idx_0_vec.data() : nullptr, use_dense_lookup ? next_idx_1_vec.data() : nullptr};
    int* prev_idxs[2] = {use_dense_lookup ? prev_idx_0_vec.data() : nullptr, use_dense_lookup ? prev_idx_1_vec.data() : nullptr};
    unsigned char* lane_ptr = lane.data();
    int* pos_ptr = pos.data();
    int* vel_ptr = vel.data();
    const bool needs_mod_slow = vmax >= road_len;

    std::vector<char> defer_ss(car_count, 0);
    std::vector<char> should_move(car_count);
    std::vector<char> deferred_grid_write(car_count);
    size_t lane_change_count_shared = 0;
    bool collect_move_list_shared = false;
    std::vector<size_t> move_count_by_thread(worker_threads_sz, 0);
    
    // Build initial lane occupancy and speed grids
    const unsigned char initial_generation = 1;
    for (size_t i = 0; i < car_count; ++i) {
        lane_ptr[i] = cars[i].lane;
        pos_ptr[i] = cars[i].position;
        vel_ptr[i] = cars[i].v;
        const int lane_i = lane_ptr[i];
        const int pos_i = pos_ptr[i];
        grid_gens[lane_i][pos_i] = initial_generation;
        write_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, lane_i, pos_i, vel_ptr[i]);
    }

    // Run all timesteps inside one parallel region
    #pragma omp parallel num_threads(worker_threads)
    {
        const int tid = omp_get_thread_num();
        const size_t tid_index = tid;
        const int nthreads = omp_get_num_threads();
        const size_t car_chunk = (car_count + nthreads - 1) / nthreads;
        const size_t start = std::min(tid_index * car_chunk, car_count);
        const size_t end = std::min(start + car_chunk, car_count);
        const size_t local_count = end - start;
        const auto prng_gap = 2 * (car_count - local_count);
        const bool has_prng_gap = (local_count > 0) && (prng_gap > 0);
        const double p_start_prob = params.p_start;
        const double p_dec_prob = params.p_dec;
        const int behind_search_limit = std::min(search_limit, vmax);
        const LaneDecisionContext lane_ctx = {lane_ptr, pos_ptr, vel_ptr, road_len, search_limit, behind_search_limit, grid_gens, use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, use_dense_lookup, next_idxs, prev_idxs};

        PRNG local_prng(params.seed);
        if (local_count > 0) {
            local_prng.discard(2 * start);
        }
        size_t grid_start = 0;
        size_t grid_end = 0;
        if (generation_may_wrap) {
            const size_t grid_chunk = (road_len_sz + nthreads - 1) / nthreads;
            grid_start = std::min(tid_index * grid_chunk, road_len_sz);
            grid_end = std::min(grid_start + grid_chunk, road_len_sz);
        }
        unsigned char generation_local = initial_generation;
        size_t prev_lane_change_count = car_count;
        std::vector<size_t> local_move_ids;
        if (local_count > 0) {
            local_move_ids.reserve(std::max<size_t>(16, local_count / 8));
        }
        for (int timestep = 0; timestep < params.steps; ++timestep) {
			// Do your calculations	
	

            // Reset per step shared counters and mode flags
            #pragma omp single
            {
                lane_change_count_shared = 0;
                collect_move_list_shared = (car_count > 0) && (prev_lane_change_count * kMoverListDivisor < car_count);
            }
            if (collect_move_list_shared) {
                local_move_ids.clear();
            }

            // Refresh dense lookup tables when enabled
            if (use_dense_lookup) {
                rebuild_dense_tables_parallel(road_len, generation_local, grid_gens, next_idxs, prev_idxs);
            }

            // Phase 1 decide lane changes for all cars
            if (use_guided_lane_decision) {
                #pragma omp for schedule(guided, 256)
                for (size_t i = 0; i < car_count; ++i) {
                    const char move_i = decide_lane_change_for_car(i, lane_ctx, generation_local);
                    should_move[i] = move_i;
                    if (collect_move_list_shared && move_i) {
                        local_move_ids.push_back(i);
                    }
                }
            } else {
                #pragma omp for schedule(static)
                for (size_t i = 0; i < car_count; ++i) {
                    const char move_i = decide_lane_change_for_car(i, lane_ctx, generation_local);
                    should_move[i] = move_i;
                    if (collect_move_list_shared && move_i) {
                        local_move_ids.push_back(i);
                    }
                }
            }
            
            // Phase 2 apply lane changes together
            if (collect_move_list_shared) {
                move_count_by_thread[tid_index] = local_move_ids.size();
                #pragma omp barrier

                #pragma omp single
                {
                    lane_change_count_shared = 0;
                    for (size_t t = 0; t < static_cast<size_t>(nthreads); ++t) {
                        lane_change_count_shared += move_count_by_thread[t];
                    }
                }
                #pragma omp barrier

                for (size_t move_idx = 0; move_idx < local_move_ids.size(); ++move_idx) {
                    const size_t i = local_move_ids[move_idx];
                    const int lane_i = lane_ptr[i];
                    const int other_lane = 1 - lane_i;
                    const int pos_i = pos_ptr[i];
                    grid_gens[lane_i][pos_i] = 0;
                    grid_gens[other_lane][pos_i] = generation_local;
                    write_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, other_lane, pos_i, vel_ptr[i]);
                }
                #pragma omp barrier
            } else {
                #pragma omp for schedule(static) reduction(+:lane_change_count_shared)
                for (size_t i = 0; i < car_count; ++i) {
                    if (should_move[i]) {
                        const int lane_i = lane_ptr[i];
                        const int other_lane = 1 - lane_i;
                        const int pos_i = pos_ptr[i];
                        grid_gens[lane_i][pos_i] = 0;
                        grid_gens[other_lane][pos_i] = generation_local;
                        write_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, other_lane, pos_i, vel_ptr[i]);
                        ++lane_change_count_shared;
                    }
                }
            }

            bool use_dense_phase_c = use_dense_lookup;
            if (use_dense_lookup && lane_change_count_shared > 0) {
                if (lane_change_count_shared >= dense_rebuild_threshold) {
                    rebuild_dense_tables_parallel(road_len, generation_local, grid_gens, next_idxs, prev_idxs);
                } else {
                    use_dense_phase_c = false;
                }
            }
            
            // Prepare next generation id and handle wrap case
            unsigned char next_generation = generation_local + 1;
            bool do_clear = false;
            if (generation_may_wrap && next_generation == 0) {
                next_generation = 1;
                do_clear = true;
            }

            // Phase 3 update velocity and position rules
            for (size_t i = start; i < end; ++i) {
                bool skip_rule2 = false;
                bool skip_rule3 = false;
                const int pos_i = pos_ptr[i];
                const int old_v = vel_ptr[i];
                int effective_lane = lane_ptr[i];

                // Draw random events for slow start and deceleration
                const bool ss_i = flip_coin(p_start_prob, &local_prng);
                const bool dec_i = flip_coin(p_dec_prob, &local_prng);
                if (should_move[i]) {
                    effective_lane = 1 - effective_lane;
                }

                // Read gap and front car speed for rule checks
                int d = kInfDistance;
                int next_vi = old_v;
                int v_ip1 = -1;
                if (use_dense_phase_c) {
                    const int hit_ahead = ahead_hit_from_table(pos_i, road_len, next_idxs[effective_lane], search_limit);
                    if (hit_ahead >= 0) {
                        d = forward_cyclic_distance(pos_i, hit_ahead, road_len);
                        v_ip1 = read_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, effective_lane, hit_ahead);
                    }
                } else {
                    d = find_ahead_distance(pos_i, road_len, grid_gens[effective_lane], search_limit, generation_local);
                    if (d != kInfDistance) {
                        int hit_pos = pos_i + d;
                        if (hit_pos >= road_len) {
                            hit_pos -= road_len;
                        }
                        v_ip1 = read_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, effective_lane, hit_pos);
                    }
                }

                // Rule 1: slow start
                if (timestep != 0 && defer_ss[i]) {
                    defer_ss[i] = false;
                    skip_rule3 = true;
                    next_vi = 1;
                } else if (next_vi == 0 && d > 1) {
                    if (ss_i) {
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
                    if (d <= next_vi && ((next_vi < v_ip1) || (next_vi < 2))) {
                        next_vi = d - 1;
                        skip_rule3 = true;

                    // case 2: the car in front is moving faster than you, or the car in front is very close to you
                    } else if (d <= next_vi && (next_vi >= v_ip1) && next_vi >= 2) {
                        next_vi = std::min(d - 1, next_vi - 2);
                        skip_rule3 = true;

                    // case 3: the car ahead is moving slower than you, and you foresee a collision in 2 timesteps
                    } else if (next_vi < d && d <= 2 * next_vi && next_vi >= v_ip1) {
                        next_vi = next_vi - (next_vi - v_ip1) / 2;
                        // Edge case assumes where (v_i - v_ip1) = 1, car velocity never changes but still counts as being modified.
                        skip_rule3 = true;
                    }
                }

                // Rule 3: acceleration
                if (!skip_rule3) {
                    next_vi = std::min(d - 1, std::min(next_vi + 1, vmax));
                }

                // Rule 4: random deceleration
                if (next_vi > 0 && dec_i) {
                    next_vi -= 1;
                }

                if (should_move[i]) {
                    // Toggle between lane 0 and 1.
                    lane_ptr[i] = 1 - lane_ptr[i];
                }

                // Rule 5: Move the car forward by its velocity units.
                int new_pos = pos_i + next_vi;
                if (new_pos >= road_len) {
                    new_pos -= road_len;
                    if (needs_mod_slow && new_pos >= road_len) {
                        new_pos %= road_len;
                    }
                }

                const int lane_i = lane_ptr[i];
                if (new_pos != pos_i) {
                    pos_ptr[i] = new_pos;
                }
                if (next_vi != old_v) {
                    vel_ptr[i] = next_vi;
                }
                const bool defer_write = do_clear || (new_pos == pos_i);
                deferred_grid_write[i] = defer_write;
                if (!defer_write) {
                    grid_gens[lane_i][new_pos] = next_generation;
                    write_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, lane_i, new_pos, next_vi);
                }
            }
            if (has_prng_gap) {
                local_prng.discard(prng_gap);
            }
            #pragma omp barrier

            if (do_clear) {
                for (size_t k = grid_start; k < grid_end; ++k) {
                    grid_gens[0][k] = 0;
                    grid_gens[1][k] = 0;
                }
                #pragma omp barrier
            }

            #pragma omp for schedule(static)
            for (size_t i = 0; i < car_count; ++i) {
                if (!deferred_grid_write[i]) {
                    continue;
                }
                const int lane_i = lane_ptr[i];
                const int pos_i = pos_ptr[i];
                grid_gens[lane_i][pos_i] = next_generation;
                write_grid_vel(use_u8_grid_vels, use_u16_grid_vels, grid_vels_u8, grid_vels_u16, grid_vels_i32, lane_i, pos_i, vel_ptr[i]);
            }

#ifdef DEBUG
            #pragma omp for schedule(static)
            for (size_t i = 0; i < car_count; ++i) {
                cars[i].lane = lane_ptr[i];
                cars[i].position = pos_ptr[i];
                cars[i].v = vel_ptr[i];
            }
            // Report the result for each timestep.
			// Make sure you update your cars vector appropriately.
            #pragma omp single
            {
                reportResult(cars, timestep);
            }
#endif
            generation_local = next_generation;
            prev_lane_change_count = lane_change_count_shared;
        }
    } // End parallel

    for (size_t i = 0; i < car_count; ++i) {
        cars[i].lane = lane_ptr[i];
        cars[i].position = pos_ptr[i];
        cars[i].v = vel_ptr[i];
    }
    // Report the final state of the cars.
    reportFinalResult(cars);
}
