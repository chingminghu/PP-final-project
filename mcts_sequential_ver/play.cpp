#include <iostream>
#include "mcts.hpp"
#include "../env/2048env.hpp"
#include "../TD_learning_sequential_ver/n_tuple_TD.hpp"

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
    
    Env2048 env;
    env.reset();
    NTupleTD agent(patterns);
    agent.load_weights("2048_weights.pkl");

    bool done = false;
    while (!done) {
        const int action = mcts_action(env.get_board(), agent, 141, 20000, 5);
        env.step(action);
        env.print_board();
        if (env.is_game_over()) {
            std::cout << "Score: " << env.get_score() << std::endl;
            done = true;
        }
    }

    return 0;
}
