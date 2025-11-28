#include "n_tuple_TD.hpp"
#include <iostream>
#include <ctime>

int main(void)
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
    
    srand(static_cast<unsigned int>(time(nullptr)));
    NTupleTD agent(patterns, 4, 4, 160000, 0.01, 1.0);
    Env2048 env;
    
    // agent.load_weights("2048_weights.pkl");
    int episodes = 100000;
    double epsilon_start = 1.0;
    double epsilon_end   = 0.05;
    int decay_episodes   = 50000;

    std::vector<int> scores = agent.train(env,
                                        episodes,
                                        epsilon_start,
                                        epsilon_end,
                                        decay_episodes);
    std::cout << "Training completed." << std::endl;

    agent.save_scores("2048_scores.txt", scores);
    agent.save_weights("2048_weights.pkl");
    std::cout << "Results saved." << std::endl;
    
    return 0;
}