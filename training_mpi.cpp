#include "n_tuple_TD.hpp"
#include <mpi.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <unordered_map>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <ctime>

using SumCountMap = std::unordered_map<Feature, std::pair<double, int>>;

std::string serialize_weights(const std::vector<WeightsMap>& weights)
{
    std::ostringstream oss;
    oss << std::setprecision(17);

    for (size_t p = 0; p < weights.size(); ++p) {
        const WeightsMap& wm = weights[p];
        for (const auto& kv : wm) {
            const Feature& feat = kv.first;
            double w = kv.second;

            oss << p << ' ' << feat.size();
            for (int v : feat) {
                oss << ' ' << v;
            }
            oss << ' ' << w << '\n';
        }
    }
    return oss.str();
}

void deserialize_weights(const std::string& data,
                         std::vector<WeightsMap>& weights,
                         size_t num_patterns)
{
    weights.clear();
    weights.resize(num_patterns);

    std::istringstream iss(data);
    size_t p_idx;
    size_t feat_len;

    while (iss >> p_idx >> feat_len) {
        if (p_idx >= num_patterns) {
            std::string dummy;
            std::getline(iss, dummy);
            continue;
        }

        Feature feat;
        feat.reserve(feat_len);
        for (size_t i = 0; i < feat_len; ++i) {
            int v;
            iss >> v;
            feat.push_back(v);
        }

        double w;
        iss >> w;
        weights[p_idx][feat] = w;
    }
}

void mpi_average_weights(NTupleTD& agent, MPI_Comm comm)
{
    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::vector<WeightsMap>& local_weights = agent.get_weights();
    const size_t num_patterns = local_weights.size();

    if (world_size == 1) {
        return;
    }

    if (rank == 0) {
        std::vector<SumCountMap> accum(num_patterns);
        auto accumulate_from = [&](const std::vector<WeightsMap>& src) {
            for (size_t p = 0; p < num_patterns; ++p) {
                const WeightsMap& wm = src[p];
                SumCountMap& ac = accum[p];
                for (const auto& kv : wm) {
                    const Feature& feat = kv.first;
                    double w = kv.second;
                    auto& entry = ac[feat];
                    entry.first  += w;
                    entry.second += 1;
                }
            }
        };

        accumulate_from(local_weights);

        for (int r = 1; r < world_size; ++r) {
            int len = 0;
            MPI_Recv(&len, 1, MPI_INT, r, 0, comm, MPI_STATUS_IGNORE);
            if (len <= 0) continue;

            std::string buf(len, '\0');
            MPI_Recv(buf.data(), len, MPI_CHAR, r, 1, comm, MPI_STATUS_IGNORE);

            std::vector<WeightsMap> tmp;
            deserialize_weights(buf, tmp, num_patterns);
            accumulate_from(tmp);
        }

        std::vector<WeightsMap> averaged(num_patterns);
        for (size_t p = 0; p < num_patterns; ++p) {
            WeightsMap& dst = averaged[p];
            for (const auto& kv : accum[p]) {
                const Feature& feat = kv.first;
                double sum = kv.second.first;
                int cnt   = kv.second.second;
                dst[feat] = sum / static_cast<double>(cnt);
            }
        }

        std::string out = serialize_weights(averaged);
        int out_len = static_cast<int>(out.size());
        MPI_Bcast(&out_len, 1, MPI_INT, 0, comm);
        if (out_len > 0) {
            MPI_Bcast(out.data(), out_len, MPI_CHAR, 0, comm);
        }

        agent.set_weights(averaged);
    } else {
        std::string data = serialize_weights(local_weights);
        int len = static_cast<int>(data.size());
        MPI_Send(&len, 1, MPI_INT, 0, 0, comm);
        if (len > 0) {
            MPI_Send(data.data(), len, MPI_CHAR, 0, 1, comm);
        }

        int out_len = 0;
        MPI_Bcast(&out_len, 1, MPI_INT, 0, comm);

        std::vector<WeightsMap> averaged;
        if (out_len > 0) {
            std::string buf(out_len, '\0');
            MPI_Bcast(buf.data(), out_len, MPI_CHAR, 0, comm);
            deserialize_weights(buf, averaged, num_patterns);
        } else {
            averaged = local_weights;
        }

        agent.set_weights(averaged);
    }

    MPI_Barrier(comm);
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::srand(static_cast<unsigned>(std::time(nullptr)) + rank * 13331);

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

    NTupleTD agent(patterns, 4, 4, 160000, 0.01, 1.0);
    Env2048 env;

    int episodes_per_proc = 25000;
    if (argc >= 2) {
        episodes_per_proc = std::atoi(argv[1]);
    }

    int sync_interval = 3000;
    if (argc >= 3) {
        sync_interval = std::atoi(argv[2]);
    }

    double epsilon_start = 1.0;
    double epsilon_end   = 0.05;
    int decay_episodes   = 16000;
    if (argc >= 4) {
        decay_episodes = std::atoi(argv[3]);
    }

    const int log_interval = 1000;
    std::vector<int> local_scores;
    local_scores.reserve(episodes_per_proc);

    auto get_epsilon = [&](int episode) -> double {
        if (episode >= decay_episodes) {
            return epsilon_end;
        }
        double ratio = static_cast<double>(episode) /
                       static_cast<double>(decay_episodes);
        return epsilon_start + ratio * (epsilon_end - epsilon_start);
    };

    for (int ep = 0; ep < episodes_per_proc; ++ep) {
        double eps = get_epsilon(ep);
        int score = agent.run_episode(env, eps);
        local_scores.push_back(score);

        if ((ep + 1) % log_interval == 0) {
            int window = std::min(log_interval,
                                  static_cast<int>(local_scores.size()));
            double local_sum = 0.0;
            for (int i = static_cast<int>(local_scores.size()) - window;
                 i < static_cast<int>(local_scores.size());
                 ++i) {
                local_sum += local_scores[i];
            }
            double local_avg = local_sum / window;

            double global_avg_sum = 0.0;
            MPI_Reduce(&local_avg, &global_avg_sum, 1,
                       MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            if (rank == 0) {
                double global_avg = global_avg_sum / world_size;
                std::cout << "[MPI] Episode (per proc): " << (ep + 1)
                          << ", epsilon: " << eps
                          << ", global avg score: " << global_avg
                          << std::endl;
            }
        }

        if ((ep + 1) % sync_interval == 0) {
            mpi_average_weights(agent, MPI_COMM_WORLD);
        }
    }

    mpi_average_weights(agent, MPI_COMM_WORLD);

    if (rank == 0) {
        agent.save_weights("2048_weights_mpi.pkl");
        std::cout << "MPI training finished. Weights saved to 2048_weights_mpi.pkl"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
