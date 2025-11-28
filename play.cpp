#include "2048env.hpp"
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
    NTupleTD agent(patterns);
    Env2048 env;
    env.reset();
    
    agent.load_weights("2048_weights_mpi.pkl");
    while(true) {
        int action = agent.choose_action(env, 0);

        env.step(action);
        env.print_board();
        
        if(env.is_game_over()) {
            std::cout << "Game Over! The final score is: " << env.get_score() << "\n";
            break;
        }
    }
}