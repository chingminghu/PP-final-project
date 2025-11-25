#include "n_tuple_TD.hpp"
#include <iostream>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <string>

int average_interval = 100;
int save_interval = 1000;

NTupleTD::NTupleTD(std::vector<Pattern>& patterns, int n_actions, int board_size, double init_value, double learning_rate, double discount_factor)
    : patterns(patterns), n_actions(n_actions), board_size(board_size), init_value(init_value), learning_rate(learning_rate), discount_factor(discount_factor)
{
    symmetric_patterns.reserve(patterns.size() * 8);
    for (const Pattern& pattern : this->patterns) {
        std::vector<Pattern> sym_patterns = generate_symmetric_patterns(pattern);
        std::copy(sym_patterns.begin(), sym_patterns.end(), std::back_inserter(symmetric_patterns));
    }
    n_tuples = symmetric_patterns.size();
    weights.resize(patterns.size());
}

std::vector<Pattern> NTupleTD::generate_symmetric_patterns(const Pattern& pattern) const
{
    Pattern p = pattern;
    std::vector<Pattern> sym_patterns;
    sym_patterns.reserve(8);
    for(int i = 0; i < 4; i++){
        sym_patterns.push_back(p);
        sym_patterns.push_back(pattern_reflect(p, board_size));
        p = pattern_rot90(p, board_size);
    }
    return sym_patterns;
}

int NTupleTD::tile_to_index(const int tile) const
{
    if (tile == 0) return 0;
    return static_cast<int>(std::log2(tile));
}

Feature NTupleTD::get_feature(const Board& board, const Pattern pattern) const
{
    Feature feature;
    for (const Coordinate& coord : pattern) {
        int y = coord.first, x = coord.second;
        feature.push_back(tile_to_index(board[y][x]));
    }
    return feature;
}

double NTupleTD::cal_value(const Board& board) //const
{
    double value = 0;
    for(int index = 0; index < n_tuples; index++){
        const Pattern& pattern = symmetric_patterns[index];
        Feature feature = get_feature(board, pattern);
        auto it = weights[index / 8].find(feature);
        if (it != weights[index / 8].end())
            value += it->second;
        else 
            value += init_value;
    }
    return value;
}

void NTupleTD::update_weights(const Board& board, const double delta)
{
    for(int index = 0; index < n_tuples; index++){
        const Pattern& pattern = symmetric_patterns[index];
        Feature feature = get_feature(board, pattern);
        auto it = weights[index / 8].find(feature);
        if (it != weights[index / 8].end())
            it->second += learning_rate * delta;
        else
            weights[index / 8][feature] = init_value + learning_rate * delta;
    }
    return;
}

double NTupleTD::simulate_action(Env2048 env, const Board& board, const int action) //const
{
    env.set_board(board);
    env.set_score(0);
    StepResult result = env.step(action);
    double value = cal_value(result.board);
    return static_cast<double>(result.score) + discount_factor * value;
}

void NTupleTD::learn(const Experience& experience)
{
    double current_value = cal_value(experience.beforestate);
    double next_value = cal_value(experience.afterstate);
    double target = static_cast<double>(experience.reward) + (experience.done ? 0 : discount_factor * next_value);
    double delta = target - current_value;
    update_weights(experience.beforestate, delta);
    return;
}

std::vector<int> NTupleTD::train(Env2048& env, const int episodes, const double epsilon)
{
    std::vector<int> scores;
    scores.reserve(episodes);

    try{
        for (int episode = 0; episode < episodes; episode++) {
            Board beforestate(board_size, Row(board_size, 0)), afterstate;
            std::vector<Experience> trajectory;
            int prev_score = 0;
            bool done = false;

            env.reset();
            while (!done){
                int action = choose_action(env, epsilon);
                if(action == -1)    break;

                StepResult result = env.step(action);
                afterstate = result.board;
                int reward = result.score - prev_score;
                done = result.game_over;

                trajectory.push_back({beforestate, action, reward, afterstate, done});
                prev_score = result.score;
                beforestate = afterstate;
            }

            for(int i = trajectory.size() - 1; i >= 0; i--) {
                Experience& exp = trajectory[i];
                learn(exp);
            }
            scores.push_back(env.get_score());
            if (episode % average_interval == 0) {
                double avg_score = std::accumulate(scores.end() - std::min(average_interval, static_cast<int>(scores.size())), scores.end(), 0.0) / std::min(average_interval, static_cast<int>(scores.size()));
                std::cout << "Episode: " << episode << ", Average Score: " << avg_score << "\n";
            }
            if (episode % save_interval == 0 && episode > 0) {
                std::cout << "Saving weights and scores at episode " << episode << "\n";
                save_weights("2048_weights.pkl");
                save_scores("2048_scores.txt", scores);
                scores.clear();
            }
        }
    }
    // catch (const std::) {
    //     std::cerr << "Training interrupted: " << e.what() << "\n";
    // }
    catch (const std::exception& e) {
        std::cerr << "Error during training: " << e.what() << "\n";
    }
    return scores;
}

int NTupleTD::choose_action(Env2048& env, const double epsilon)
{
    std::vector<int> legal_actions = env.get_legal_actions();
    if (legal_actions.empty()) return -1;
    if (static_cast<double>(rand()) / RAND_MAX < epsilon)
        return legal_actions[rand() % legal_actions.size()];

    std::vector<double> action_values(n_actions, -std::numeric_limits<double>::infinity());
    for (int action : legal_actions) {
        action_values[action] = simulate_action(env, env.get_board(), action);
    }
    return std::distance(action_values.begin(), std::max_element(action_values.begin(), action_values.end()));
}

void NTupleTD::save_scores(const std::string& path, const std::vector<int>& scores) const
{
    std::ofstream ofs(path, std::ios_base::app);
    if(!ofs.is_open()) {
        std::cerr << "Error opening file for saving scores: " << path << "\n";
        return;
    }
    for(int score : scores) {
        ofs << score << "\n";
    }
    ofs.close();
    std::cout << "Scores saved to " << path << "\n";
}

void NTupleTD::save_weights(const std::string& path) const
{
    std::ofstream ofs(path);
    if(!ofs.is_open()) {
        std::cerr << "Error opening file for saving weights: " << path << "\n";
        return;
    }

    for(int i = 0; i < weights.size(); i++) {
        const WeightsMap& weight_map = weights[i];
        ofs << "Pattern " << i << ":\n";
        for(const auto& pair : weight_map) {
            for(int feature : pair.first) {
                ofs << feature << " ";
            }
            ofs << "; " << pair.second << "\n";
        }
    }
    ofs.close();
    std::cout << "Weights saved to " << path << "\n";
    return;
}

void NTupleTD::load_weights(const std::string& path)
{
    std::ifstream ifs(path);
    if(!ifs.is_open()) {
        std::cerr << "Error opening file for loading weights: " << path << "\n";
        return;
    }
    weights.clear();
    weights.resize(patterns.size());
    std::string line;
    int pattern_index = -1;
    while(std::getline(ifs, line)) {
        if(line.empty() || line[0] == '#') continue;
        if(line.find("Pattern") != std::string::npos) {
            pattern_index = std::stoi(line.substr(line.find(" ") + 1));
            if(pattern_index < 0 || pattern_index >= patterns.size()) {
                std::cerr << "Invalid pattern index in weights file: " << pattern_index << "\n";
                continue;
            }
            weights[pattern_index].clear();
        } 
        else {
            std::istringstream iss(line);
            Feature feature;
            double weight;
            int value;

            while (iss >> value) {
                feature.push_back(value);
                if (iss >> std::ws && iss.peek() == ';') {
                    iss.get();
                    break;
                }
            }
            iss >> weight;
            weights[pattern_index][feature] = weight;
        }
    }
    ifs.close();
    std::cout << "Weights loaded from " << path << "\n";
    return;
}

Pattern pattern_rot90(const Pattern& pattern, const int board_size)
{
    Pattern rotated;
    for (const Coordinate& coord : pattern) {
        int new_y = coord.second;
        int new_x = board_size - 1 - coord.first;
        rotated.emplace_back(std::make_pair(new_y, new_x));
    }
    return rotated;
}

Pattern pattern_reflect(const Pattern& pattern, const int board_size)
{
    Pattern reflected;
    for (const Coordinate& coord : pattern) {
        int new_y = coord.first;
        int new_x = board_size - 1 - coord.second;
        reflected.emplace_back(std::make_pair(new_y, new_x));
    }
    return reflected;
}