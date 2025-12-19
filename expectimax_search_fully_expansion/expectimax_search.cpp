#include "2048env.hpp"
#include "n_tuple_TD.hpp"
#include "expectimax_search.hpp"
#include "thread_pool.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <thread>
#include <future>
#include <iostream>

ThreadPool thread_pool(8);

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
    std::vector<std::future<double>> future_results;
    if(is_maxNode){
        // Max Node
        value = -std::numeric_limits<double>::infinity();
        std::vector<double> rewards;
        std::vector<int> actions = env.get_legal_actions();
        for(int action: actions){
            env.set_board(state);
            env.set_score(0);
            auto [next_state, reward, done] = env.step(action);
            rewards.push_back(static_cast<double>(reward));
            future_results.emplace_back(std::async(std::launch::async,
                [=, &agent] {
                    return expectimax_search(next_state, agent,
                                                   depth - 1, num_sample, false);
                }
            ));
        }

        for(int i = 0; i < future_results.size(); i++){
            value = std::max(value, rewards[i] + future_results[i].get());
        }
    }
    else{
        // Chance Node
        for(int i = 0; i < num_sample; i++){
            env.set_board(state);
            env.set_score(0);
            env.add_random_tile();
            Board next_state = env.get_board();
            future_results.emplace_back(std::async(std::launch::async,
                [=, &agent] {
                    return expectimax_search(next_state, agent,
                                                   depth - 1, num_sample, true);
                }
            ));
        }

        for(auto &future: future_results){
            value += future.get();
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

    std::vector<std::future<double>> future_results;
    std::vector<double> action_values;
    for (int action: actions) {
        env.set_board(root);
        env.set_score(0);
        auto [next_state, reward, done] = env.step(action);
        action_values.push_back(static_cast<double>(reward));
        future_results.emplace_back(std::async(std::launch::async,
            [=, &agent] {
                return expectimax_search(next_state, agent,
                                                depth - 1, num_sample, false);
            }
        ));
    }

    for(int i = 0; i < future_results.size(); i++) {
        action_values[i] += future_results[i].get();
    }

    int index = std::distance(action_values.begin(), std::max_element(action_values.begin(), action_values.end()));
    return actions[index];
}