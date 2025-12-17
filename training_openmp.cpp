#include "n_tuple_TD.hpp"
#include <omp.h> // Replaced mpi.h with omp.h

#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <cstring>

// Logic for averaging weights (Sum of weights, Count of contributors)
using SumCountMap = std::unordered_map<Feature, std::pair<double, int>>;

// Training curve logging entry
struct TrainingLogEntry {
    int episode;        // Episode number (per thread)
    double epsilon;     // Epsilon used at this episode
    double global_avg;  // Global average score across threads
    double time_sec;    // Wall-clock time since start (seconds)
};

// Helper to merge local weights into a global accumulator (Thread-safe via critical section in main)
void accumulate_weights(const std::vector<WeightsMap>& local_weights, 
                        std::vector<SumCountMap>& global_accum) 
{
    for (size_t p = 0; p < local_weights.size(); ++p) {
        const WeightsMap& wm = local_weights[p];
        SumCountMap& ac = global_accum[p];
        for (const auto& kv : wm) {
            const Feature& feat = kv.first;
            double w = kv.second;
            
            auto& entry = ac[feat];
            entry.first  += w;
            entry.second += 1;
        }
    }
}

// Helper to calculate averages from the accumulator
void compute_averages(const std::vector<SumCountMap>& global_accum, 
                      std::vector<WeightsMap>& global_averaged) 
{
    for (size_t p = 0; p < global_accum.size(); ++p) {
        WeightsMap& dst = global_averaged[p];
        dst.clear(); // Clear previous averages
        for (const auto& kv : global_accum[p]) {
            const Feature& feat = kv.first;
            double sum = kv.second.first;
            int cnt   = kv.second.second;
            dst[feat] = sum / static_cast<double>(cnt);
        }
    }
}

int main(int argc, char** argv)
{
    // 1. Setup arguments and parameters
    int episodes_per_thread = 25000;// 25000;
    if (argc >= 2) episodes_per_thread = std::atoi(argv[1]);

    int sync_interval = 5000;//3000;
    if (argc >= 3) sync_interval = std::atoi(argv[2]);

    double epsilon_start = 1.0;
    double epsilon_end   = 0.05;
    int decay_episodes   = static_cast<int>(episodes_per_thread * 0.9);// 16000;
    if (argc >= 4) decay_episodes = std::atoi(argv[3]);

    const int log_interval = 1000;

    // Log file name (customizable via argv[4], default provided)
    std::string log_filename = "training_log_omp.json";
    if (argc >= 5 && argv[4] && std::strlen(argv[4]) > 0) {
        log_filename = argv[4];
    }

    // 2. Define N-Tuple Patterns
    std::vector<Pattern> patterns = {
        {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 1}, {1, 2}},
        {{0, 1}, {0, 2}, {1, 1}, {1, 2}, {2, 1}, {3, 1}},
        {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}},
        {{0, 0}, {0, 1}, {1, 1}, {1, 2}, {1, 3}, {2, 2}},
        {{0, 0}, {0, 1}, {0, 2}, {1, 1}, {2, 1}, {2, 2}},
        {{0, 0}, {0, 1}, {1, 1}, {2, 1}, {3, 1}, {3, 2}},
        {{0, 0}, {0, 1}, {1, 1}, {2, 0}, {2, 1}, {3, 1}},
        {{0, 0}, {0, 1}, {0, 2}, {1, 0}, {1, 2}, {2, 2}}
    };

    // Shared data structures for synchronization
    std::vector<SumCountMap> global_accum(patterns.size());
    std::vector<WeightsMap> global_averaged(patterns.size());
    std::vector<TrainingLogEntry> training_log; // Shared training curve records
    
    // Timer for performance checking (optional)
    double start_time = omp_get_wtime();

    std::cout << "Starting OpenMP Training..." << std::endl;

    std::cout << omp_get_num_procs() << " processors detected." << std::endl;

    // 3. Start Parallel Region
    #pragma omp parallel num_threads(omp_get_num_procs())
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();


        #pragma omp critical
        std::cout << "Thread " << thread_id << " of " << num_threads << " started." << std::endl;

        // Seed RNG differently per thread
        std::srand(static_cast<unsigned>(std::time(nullptr)) + thread_id * 13331);

        // Thread-local Agent and Environment
        NTupleTD agent(patterns, 4, 4, 160000, 0.01, 1.0);
        Env2048 env;
        
        // Local score tracking
        std::vector<int> local_scores;
        local_scores.reserve(episodes_per_thread);

        // Epsilon decay lambda
        auto get_epsilon = [&](int episode) -> double {
            if (episode >= decay_episodes) return epsilon_end;
            double ratio = static_cast<double>(episode) / static_cast<double>(decay_episodes);
            return epsilon_start + ratio * (epsilon_end - epsilon_start);
        };

        // Training Loop
        for (int ep = 0; ep < episodes_per_thread; ++ep) {
            double eps = get_epsilon(ep);
            int score = agent.run_episode(env, eps);
            local_scores.push_back(score);

            // --- Logging ---
            if ((ep + 1) % log_interval == 0) {
                // Calculate local average
                int window = std::min(log_interval, static_cast<int>(local_scores.size()));
                double local_sum = 0.0;
                for (int i = static_cast<int>(local_scores.size()) - window; i < static_cast<int>(local_scores.size()); ++i) {
                    local_sum += local_scores[i];
                }
                double local_avg = local_sum / window;

                // Combine logs safely
                // We use a critical section or barrier + master print. 
                // To keep it clean like the MPI version, we sum all averages then print once.
                
                static double global_score_sum = 0.0; // Static shared inside parallel
                
                #pragma omp barrier // Wait for all to finish episode batch
                
                #pragma omp single
                global_score_sum = 0.0; // Reset
                
                #pragma omp critical
                global_score_sum += local_avg;

                #pragma omp barrier // Wait for addition

                #pragma omp single
                {
                    double global_avg = global_score_sum / num_threads;
                    // Compute epsilon deterministically here using shared params
                    double eps_single;
                    if (ep >= decay_episodes) {
                        eps_single = epsilon_end;
                    } else {
                        double ratio = static_cast<double>(ep) / static_cast<double>(decay_episodes);
                        eps_single = epsilon_start + ratio * (epsilon_end - epsilon_start);
                    }
                    double elapsed = omp_get_wtime() - start_time;
                    // Record training curve point
                    training_log.push_back(TrainingLogEntry{ep + 1, eps_single, global_avg, elapsed});
                    std::cout << "[OpenMP] Episode (per thread): " << (ep + 1)
                              << ", epsilon: " << eps
                              << ", global avg score: " << global_avg 
                              << " (Threads: " << num_threads << ")" << std::endl;
                }
            }

            // --- Weight Synchronization ---
            if ((ep + 1) % sync_interval == 0) {
                #pragma omp barrier // 1. Stop all threads

                #pragma omp single
                {
                    // 2. Master clears the global accumulator
                    for(auto& map : global_accum) map.clear();
                }
                // Implicit barrier after single

                // 3. Each thread merges its weights into global_accum
                // Note: merging maps is not thread safe, so we need critical
                #pragma omp critical
                {
                    accumulate_weights(agent.get_weights(), global_accum);
                }
                
                #pragma omp barrier // 4. Wait for all merges

                #pragma omp single
                {
                    // 5. Compute the new average
                    compute_averages(global_accum, global_averaged);
                }
                // Implicit barrier

                // 6. Update local agent with new global weights
                agent.set_weights(global_averaged);
                
                // 7. Continue training
            }
        }

        // --- Final Synchronization at end of training ---
        #pragma omp barrier
        #pragma omp single
        for(auto& map : global_accum) map.clear();

        #pragma omp critical
        accumulate_weights(agent.get_weights(), global_accum);

        #pragma omp barrier
        #pragma omp single
        {
            compute_averages(global_accum, global_averaged);
            
            // Save weights (Master only)
            // Assuming the agent class has a helper or we reuse the local agent of thread 0
            // We set thread 0's agent to the final result to save it.
            // Write training curve JSON once (master only)
            std::ofstream ofs(log_filename);
            ofs << "{\n  \"log\": [\n";
            for (size_t i = 0; i < training_log.size(); ++i) {
                const auto &e = training_log[i];
                ofs << "    {\"episode\": " << e.episode
                    << ", \"epsilon\": " << std::fixed << std::setprecision(6) << e.epsilon
                    << ", \"global_avg\": " << std::fixed << std::setprecision(6) << e.global_avg
                    << ", \"time_sec\": " << std::fixed << std::setprecision(6) << e.time_sec
                    << "}";
                if (i + 1 < training_log.size()) ofs << ",";
                ofs << "\n";
            }
            ofs << "  ]\n}\n";
            ofs.close();
        }
        
        // Final update to local agents (needed if we continued, but we are done)
        if (thread_id == 0) {
            agent.set_weights(global_averaged);
            agent.save_weights("2048_weights_omp.pkl");
            std::cout << "OpenMP training finished. Time: " << (omp_get_wtime() - start_time) << "s" << 
                         ", log saved to: " << log_filename << std::endl;
        }
    } // End Parallel

    return 0;
}