#include "2048env.hpp"
#include "n_tuple_TD.hpp"
#include "expectimax_search.hpp"
#include <iostream>
#include <chrono>

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
    
    NTupleTD agent(patterns);
    Env2048 env;
    env.reset();
    
    agent.load_weights("2048_weights.pkl");
    std::chrono::duration<double, std::milli> duration;
    double total_duration;
    int n_step = 0;
    while(true) {
        auto start = std::chrono::high_resolution_clock::now();
        int action = Expectimax(env.get_board(), agent, 5, 20);
        auto end = std::chrono::high_resolution_clock::now();
        duration = end - start;
        if(n_step < 100){
            total_duration += duration.count();
        }
        n_step += 1;

        env.step(action);
        env.print_board();
        
        if(env.is_game_over()) {
            std::cout << "Game Over! The final score is: " << env.get_score() << "\n";
            break;
        }
    }
    std::cout << "The average time spent for steps is " << total_duration / 100 << std::endl; 
}