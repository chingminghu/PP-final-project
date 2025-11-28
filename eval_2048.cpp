#include "n_tuple_TD.hpp"
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <ctime>

std::vector<Pattern> build_patterns() {
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
    return patterns;
}

int eval_one_episode(NTupleTD &agent, Env2048 &env, double epsilon = 0.0) {
    env.reset();
    bool done = false;

    while (!done) {
        int action = agent.choose_action(env, epsilon);
        if (action == -1) {
            break;
        }

        StepResult result = env.step(action);
        done = result.game_over;
    }

    return env.get_score();
}

int main(int argc, char* argv[]) {
    int eval_episodes = 1000;
    std::string weight_path = "2048_weights_mpi.pkl";

    if (argc >= 2) {
        eval_episodes = std::stoi(argv[1]);
    }
    if (argc >= 3) {
        weight_path = argv[2];
    }

    std::vector<Pattern> patterns = build_patterns();

    srand(static_cast<unsigned int>(time(nullptr)));
    NTupleTD agent(patterns, 4, 4, 160000, 0.01, 1.0);

    agent.load_weights(weight_path);

    Env2048 env;

    std::vector<int> scores;
    scores.reserve(eval_episodes);

    for (int ep = 0; ep < eval_episodes; ++ep) {
        int score = eval_one_episode(agent, env, 0.0);
        scores.push_back(score);
    }

    double sum = std::accumulate(scores.begin(), scores.end(), 0.0);
    double avg = (eval_episodes > 0) ? (sum / eval_episodes) : 0.0;

    int min_score = *std::min_element(scores.begin(), scores.end());
    int max_score = *std::max_element(scores.begin(), scores.end());

    std::cout << "====================================\n";
    std::cout << "Eval episodes: " << eval_episodes << "\n";
    std::cout << "Average score: " << avg << "\n";
    std::cout << "Min score:     " << min_score << "\n";
    std::cout << "Max score:     " << max_score << "\n";
    std::cout << "====================================\n";

    return 0;
}
