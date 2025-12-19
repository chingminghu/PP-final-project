#ifndef EXPECTIMAX_SEARCH_HPP
#define EXPECTIMAX_SEARCH_HPP

#include "2048env.hpp"
#include "n_tuple_TD.hpp"

#define DEFAULT_NUM_SAMPLE 10

int Expectimax(const Board &root, const NTupleTD &agent, int depth, int num_sample = DEFAULT_NUM_SAMPLE);

#endif