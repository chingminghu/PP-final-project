#include "n_tuple_TD.hpp"
#include <iostream>

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
    
    NTupleTD agent(patterns, 4, 4, 160000, 0.01, 1.0);
    Env2048 env;
    
    agent.load_weights("2048_weights.pkl");
    std::vector<int> scores = agent.train(env, 1000000, 1.0);
    std::cout << "Training completed.\n";

    // std::cout << "Save the results? (y/n): ";
    // char choice;
    // std::cin >> choice;
    // if (choice == 'y' || choice == 'Y') {
    //     agent.save_scores("2048_scores.txt", scores);
    //     agent.save_weights("2048_weights.pkl");
    // } 
    // else {
    //     std::cout << "Results not saved.\n";
    // }
    
    return 0;
}