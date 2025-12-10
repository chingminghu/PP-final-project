#pragma once
#include "../env/2048env.hpp"
#include "../TD_learning_sequential_ver/n_tuple_TD.hpp"

int mcts_action(const Board &board, const NTupleTD &agent,
                const double exploration_constant = 1.41,
                const int iterations = 500, const int rollout_depth = 10);
