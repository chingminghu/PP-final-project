#include "2048env.hpp"
#include "n_tuple_TD.hpp"
#include "expectimax_search.hpp"

#include <vector>
#include <limits>
#include <algorithm>

double heuristic(const Board &state, const NTupleTD &agent){
    return agent.cal_value(state);
}

double expectimax_search(const Board &state, const NTupleTD &agent, int depth, int num_sample, bool is_maxNode){
    if(depth == 0){
        return heuristic(state, agent);
    }

    Env2048 env;
    env.set_board(state);
    if(env.is_game_over()){
        return heuristic(state, agent);
    }

    double value = 0;
    if(is_maxNode){
        // Max Node
        value = -std::numeric_limits<double>::infinity();
        std::vector<int> actions = env.get_legal_actions();
        for(int action: actions){
            env.set_board(state);
            env.set_score(0);
            auto [next_state, reward, done] = env.step(action);
            double score = static_cast<double>(reward) +
                        expectimax_search(next_state, agent, depth - 1, num_sample, false);
            if(score > value){
                value = score;
            }
        }
    }
    else{
        // Chance Node
        for(int i = 0; i < num_sample; i++){
            env.set_board(state);
            env.set_score(0);
            env.add_random_tile();
            value += expectimax_search(env.get_board(), agent, depth - 1, num_sample, true);
        }
        value /= num_sample;
    }
    return value;
}

int Expectimax(const Board &root, const NTupleTD &agent, int depth, int num_sample){
    if(depth <= 0 || num_sample <= 0){
        return -1;
    }

    Env2048 env;
    env.set_board(root);
    std::vector<int> actions = env.get_legal_actions();
    if(actions.size() == 0){
        return -1;
    }

    std::vector<double> action_values(env.get_n_actions(), -std::numeric_limits<double>::infinity());
    for (int action : actions) {
        env.set_board(root);
        env.set_score(0);
        auto [next_state, reward, done] = env.step(action);
        action_values[action] = static_cast<double>(reward) +
                                expectimax_search(next_state, agent, depth - 1, num_sample, false);
    }
    return std::distance(action_values.begin(), std::max_element(action_values.begin(), action_values.end()));
}