#include "2048env.hpp"
#include "n_tuple_TD.hpp"
#include "expectimax_search.hpp"
#include "thread_pool.hpp"

#include <vector>
#include <limits>
#include <algorithm>
#include <thread>

ThreadPool thread_pool(std::thread::hardware_concurrency() - 1);

double heuristic(const Board &state, const NTupleTD &agent){
    return agent.cal_value(state);
}

double expectimax_search(const Board &state, const NTupleTD &agent, int depth, int num_sample, bool is_maxNode){
    if(depth <= 0){
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

void worker_func(const Board& state, int action, const NTupleTD& agent, int depth, int num_sample, double& result){
    Env2048 env;
    env.set_board(state);
    env.set_score(0);
    auto [next_state, reward, done] = env.step(action);
    result = static_cast<double>(reward) +
             expectimax_search(next_state, agent, depth - 1, num_sample, false);
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
    std::vector<double> rewards(env.get_n_actions());
    std::vector<std::vector<std::future<double>>> future_results;
    future_results.resize(env.get_n_actions());
    for (int action : actions) {
        env.set_board(root);
        env.set_score(0);
        auto [next_state, reward, done] = env.step(action);
        rewards[action] = reward;

        env.set_board(next_state);
        if(env.is_game_over()){
            action_values[action] = reward + heuristic(next_state, agent);
            continue;
        }

        for(int i = 0; i < num_sample; i++){
            env.set_board(next_state);
            env.set_score(0);
            env.add_random_tile();
            const Board &next_state = env.get_board();
            future_results[action].push_back(thread_pool.submit([=, &agent]{
                return expectimax_search(next_state, agent, depth - 2, num_sample, true);
            }));
        }
    }

    for(int action : actions){
        if(action_values[action] != -std::numeric_limits<double>::infinity()){
            continue;
        }
        action_values[action] = 0;
        for(auto &f : future_results[action]){
            action_values[action] += f.get();
        }
        action_values[action] /= num_sample;
        action_values[action] += rewards[action];
    }
    return std::distance(action_values.begin(), std::max_element(action_values.begin(), action_values.end()));
}