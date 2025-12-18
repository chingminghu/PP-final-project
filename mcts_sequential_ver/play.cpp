#include <iostream>
#include "mcts.hpp"
#include "../env/2048env.hpp"
#include "../TD_learning_sequential_ver/n_tuple_TD.hpp"
#include <chrono>
#include <iomanip>

int main()
{
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

    NTupleTD agent(patterns);
    agent.load_weights("2048_weights.pkl");
    Env2048 env;
    env.reset();
    bool done = false;
    unsigned long long total_time = 0, total_step = 0, time_100 = 0;
    while (!done) {
        std::chrono::steady_clock::time_point t_begin = std::chrono::steady_clock::now();
        const int action = mcts_action(env.get_board(), agent, 1.41, 4096, 5);
        // const int action = agent.choose_action(env);
        std::chrono::steady_clock::time_point t_end = std::chrono::steady_clock::now();
        unsigned long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_begin).count();
        total_time += duration;
        if (total_step < 100)
            time_100 += duration;
        env.step(action);
        env.print_board();
        if (env.is_game_over()) {
            done = true;
        }
        total_step++;
    }
    std::cout << "Score: " << env.get_score() << std::endl;
    std::cout << "Average time for one step (first 100 steps): " << (double)time_100 / 100 << " (ms)" << std::endl;
    std::cout << "Average time for one step (whole game): " << (double)total_time / total_step << " (ms)" << std::endl;

    return 0;
}
