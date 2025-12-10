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

    NTupleTD agent(patterns);
    agent.load_weights("2048_weights.pkl");
    Env2048 env;
    env.reset();
    bool done = false;
    while (!done) {
        const int action = mcts_action(env.get_board(), agent, 1.41, 2000, 5);
        // const int action = agent.choose_action(env, 0);
        env.step(action);
        env.print_board();
        if (env.is_game_over()) {
            done = true;
        }
    }
    std::cout << "Score: " << env.get_score() << std::endl;

    return 0;
}
