#include "n_tuple_TD.hpp"
#include <mpi.h>

#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <cstring>

using SumCountMap = std::unordered_map<Feature, std::pair<double, int>>;
using Byte        = unsigned char;

double g_mpi_average_time_local = 0.0;

static void pack_int(std::vector<Byte>& buf, int value)
{
    Byte bytes[sizeof(int)];
    std::memcpy(bytes, &value, sizeof(int));
    buf.insert(buf.end(), bytes, bytes + sizeof(int));
}

static void pack_double(std::vector<Byte>& buf, double value)
{
    Byte bytes[sizeof(double)];
    std::memcpy(bytes, &value, sizeof(double));
    buf.insert(buf.end(), bytes, bytes + sizeof(double));
}

static bool can_read(size_t offset, size_t need, size_t len)
{
    return offset + need <= len;
}

static int unpack_int(const Byte* data, size_t& offset, size_t len)
{
    int value = 0;
    if (!can_read(offset, sizeof(int), len)) {
        return 0;
    }
    std::memcpy(&value, data + offset, sizeof(int));
    offset += sizeof(int);
    return value;
}

static double unpack_double(const Byte* data, size_t& offset, size_t len)
{
    double value = 0.0;
    if (!can_read(offset, sizeof(double), len)) {
        return 0.0;
    }
    std::memcpy(&value, data + offset, sizeof(double));
    offset += sizeof(double);
    return value;
}

static std::vector<Byte> serialize_weights_binary(
    const std::vector<WeightsMap>& weights)
{
    std::vector<Byte> buf;

    for (size_t p = 0; p < weights.size(); ++p) {
        const WeightsMap& wm = weights[p];
        for (const auto& kv : wm) {
            const Feature& feat = kv.first;
            double w            = kv.second;

            pack_int(buf, static_cast<int>(p));
            pack_int(buf, static_cast<int>(feat.size()));
            for (int v : feat) {
                pack_int(buf, v);
            }
            pack_double(buf, w);
        }
    }
    return buf;
}

static void deserialize_weights_binary(
    const Byte* data,
    size_t len,
    size_t num_patterns,
    std::vector<WeightsMap>& weights)
{
    weights.clear();
    weights.resize(num_patterns);

    size_t offset = 0;
    while (can_read(offset, 2 * sizeof(int), len)) {
        int p_idx    = unpack_int(data, offset, len);
        int feat_len = unpack_int(data, offset, len);

        if (feat_len < 0) {
            break;
        }

        size_t need = static_cast<size_t>(feat_len) * sizeof(int) + sizeof(double);
        if (!can_read(offset, need, len)) {
            break;
        }

        Feature feat;
        feat.reserve(static_cast<size_t>(feat_len));
        for (int i = 0; i < feat_len; ++i) {
            int v = unpack_int(data, offset, len);
            feat.push_back(v);
        }
        double w = unpack_double(data, offset, len);

        if (p_idx < 0 || static_cast<size_t>(p_idx) >= num_patterns) {
            continue;
        }
        weights[static_cast<size_t>(p_idx)][std::move(feat)] = w;
    }
}

void mpi_average_weights(NTupleTD& agent, MPI_Comm comm)
{
    double t0 = MPI_Wtime();
    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    const std::vector<WeightsMap>& local_weights = agent.get_weights();
    const size_t num_patterns = local_weights.size();

    if (world_size == 1) {
        return;
    }

    std::vector<Byte> local_buf = serialize_weights_binary(local_weights);
    int local_size = static_cast<int>(local_buf.size());

    std::vector<int> all_sizes(world_size);
    MPI_Allgather(&local_size, 1, MPI_INT,
                  all_sizes.data(), 1, MPI_INT,
                  comm);

    std::vector<int> displs(world_size);
    int total_size = 0;
    for (int i = 0; i < world_size; ++i) {
        displs[i] = total_size;
        total_size += all_sizes[i];
    }

    std::vector<Byte> all_buf(static_cast<size_t>(total_size));
    MPI_Allgatherv(local_buf.data(), local_size, MPI_UNSIGNED_CHAR,
                   all_buf.data(), all_sizes.data(), displs.data(),
                   MPI_UNSIGNED_CHAR, comm);

    std::vector<SumCountMap> accum(num_patterns);

    for (int r = 0; r < world_size; ++r) {
        int sz = all_sizes[r];
        if (sz <= 0) continue;

        size_t start = static_cast<size_t>(displs[r]);
        size_t len   = static_cast<size_t>(sz);

        std::vector<WeightsMap> tmp;
        deserialize_weights_binary(all_buf.data() + start, len,
                                   num_patterns, tmp);

        for (size_t p = 0; p < num_patterns; ++p) {
            const WeightsMap& wm = tmp[p];
            SumCountMap& ac      = accum[p];
            for (const auto& kv : wm) {
                const Feature& feat = kv.first;
                double w            = kv.second;
                auto& entry = ac[feat];
                entry.first  += w;
                entry.second += 1;
            }
        }
    }

    std::vector<WeightsMap> averaged(num_patterns);
    for (size_t p = 0; p < num_patterns; ++p) {
        WeightsMap& dst = averaged[p];
        for (const auto& kv : accum[p]) {
            const Feature& feat = kv.first;
            double sum          = kv.second.first;
            int cnt             = kv.second.second;
            dst[feat] = sum / static_cast<double>(cnt);
        }
    }

    agent.set_weights(averaged);

    MPI_Barrier(comm);
    double t1 = MPI_Wtime();
    g_mpi_average_time_local += (t1 - t0);
}

double mpi_evaluate_agent(NTupleTD& agent,
                          int eval_episodes_per_proc,
                          double epsilon_eval,
                          MPI_Comm comm)
{
    int rank, world_size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &world_size);

    Env2048 eval_env;
    double local_sum = 0.0;

    for (int i = 0; i < eval_episodes_per_proc; ++i) {
        int score = agent.run_episode(eval_env, epsilon_eval);
        local_sum += static_cast<double>(score);
    }

    double global_sum = 0.0;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, comm);

    int total_episodes = eval_episodes_per_proc * world_size;
    double avg_score = 0.0;
    if (rank == 0 && total_episodes > 0) {
        avg_score = global_sum / static_cast<double>(total_episodes);
    }

    MPI_Bcast(&avg_score, 1, MPI_DOUBLE, 0, comm);

    return avg_score;
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

    int train_episodes_per_proc = 25000;
    if (argc >= 2) {
        train_episodes_per_proc = std::atoi(argv[1]);
    }

    int sync_interval = 1000;
    if (argc >= 3) {
        sync_interval = std::atoi(argv[2]);
    }

    double epsilon_start = 1.0;
    double epsilon_end   = 0.05;
    int decay_episodes   = 20000;
    if (argc >= 4) {
        decay_episodes = std::atoi(argv[3]);
    }

    int eval_episodes_per_proc = 100;
    if (argc >= 5) {
        eval_episodes_per_proc = std::atoi(argv[4]);
    }

    const int log_interval = 1000;
    double train_start = MPI_Wtime();

    auto get_epsilon = [&](int episode) -> double {
        if (episode >= decay_episodes) {
            return epsilon_end;
        }
        double ratio = static_cast<double>(episode) /
                       static_cast<double>(decay_episodes);
        return epsilon_start + ratio * (epsilon_end - epsilon_start);
    };

    for (int ep = 0; ep < train_episodes_per_proc; ++ep) {
        double eps = get_epsilon(ep);
        agent.run_episode(env, eps);

        if ((ep + 1) % sync_interval == 0 || ep == train_episodes_per_proc - 1) {
            mpi_average_weights(agent, MPI_COMM_WORLD);
        }

        if ((ep + 1) % log_interval == 0) {
            double elapsed = MPI_Wtime() - train_start;

            double eval_avg = mpi_evaluate_agent(
                agent,
                eval_episodes_per_proc,
                0.0,
                MPI_COMM_WORLD
            );

            int local_episodes  = ep + 1;

            if (rank == 0) {
                std::cout << "[LOG] time=" << elapsed
                          << "s, world_size=" << world_size
                          << ", ep_per_proc=" << local_episodes
                          << ", epsilon=" << eps
                          << ", eval_avg_score=" << eval_avg
                          << std::endl;
            }
        }
    }

    double total_time = 0.0;
    double max_time   = 0.0;
    MPI_Reduce(&g_mpi_average_time_local, &total_time, 1,
               MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&g_mpi_average_time_local, &max_time, 1,
               MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double avg_time_per_rank = total_time / world_size;
        std::cout << "[PROFILE] mpi_average_weights total time per rank (sum): "
                  << total_time << " s, avg per rank: " << avg_time_per_rank
                  << " s, max rank time: " << max_time << " s"
                  << std::endl;
    }

    if (rank == 0) {
        agent.save_weights("2048_weights_mpi.pkl");
        std::cout << "MPI training finished. Weights saved to 2048_weights_mpi.pkl"
                  << std::endl;
    }

    MPI_Finalize();
    return 0;
}
