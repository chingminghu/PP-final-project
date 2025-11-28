#ifndef N_TUPLE_TD_HPP
#define N_TUPLE_TD_HPP

#include "2048env.hpp"
#include <unordered_map>
#include <utility>
#include <string>

typedef std::pair<int, int> Coordinate;
typedef std::vector<Coordinate> Pattern;

typedef std::vector<int> Feature;
typedef std::unordered_map<Feature, double> WeightsMap;

namespace std {
    template <>
    struct hash<std::vector<int>> {
        size_t operator()(const std::vector<int>& v) const {
            size_t h = 0;
            for (int x : v)
                h ^= std::hash<int>{}(x) + 0x9e3779b9 + (h << 6) + (h >> 2);  // boost hash_combine
            return h;
        }
    };
}

typedef struct {
    Board beforestate;
    int action;
    int reward;
    Board afterstate;
    bool done;
} Experience;

class NTupleTD
{
private:
    std::vector<Pattern> patterns;
    int n_tuples;
    int n_actions;
    int board_size;
    double learning_rate;
    double discount_factor;
    double init_value;
    std::vector<Pattern> symmetric_patterns;
    std::vector<WeightsMap> weights;

    std::vector<Pattern> generate_symmetric_patterns(const Pattern& pattern) const;
    int tile_to_index(const int tile) const;
    Feature get_feature(const Board& board, const Pattern pattern) const;
    double cal_value(const Board& board);
    void update_weights(const Board& board, const double delta);
    double simulate_action(Env2048 env, const Board& board, const int action);
    void learn(const Experience& experience);

public:
    NTupleTD(std::vector<Pattern>& patterns,
             int n_actions = 4,
             int board_size = 4,
             double init_value = 0.0,
             double learning_rate = 0.01,
             double discount_factor = 0.99);

    std::vector<int> train(Env2048& env,
                           const int episodes = 10000,
                           const double epsilon_start = 0.1,
                           double epsilon_end = 0.1,
                           int decay_episodes = 10000);
    int choose_action(Env2048& env, const double epsilon = 0.1);
    void save_scores(const std::string& path, const std::vector<int>& scores) const;
    void save_weights(const std::string& path) const;
    void load_weights(const std::string& path);
    int run_episode(Env2048& env, double epsilon);
    const std::vector<WeightsMap>& get_weights() const;
    void set_weights(const std::vector<WeightsMap>& new_weights);
};

Pattern pattern_rot90(const Pattern& pattern, const int board_size);
Pattern pattern_reflect(const Pattern& pattern, const int board_size);

#endif