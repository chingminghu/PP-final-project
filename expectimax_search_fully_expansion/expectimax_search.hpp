#ifndef EXPECTIMAX_SEARCH_HPP
#define EXPECTIMAX_SEARCH_HPP

#include "2048env.hpp"
#include "n_tuple_TD.hpp"

#include <vector>
#include <queue>

#define DEFAULT_NUM_SAMPLE 10

class Node
{
    public:
        Board state;
        double value;
        bool is_leaf;       // If depth == 0 or game over
        int depth;          // Remaining depth
        Node *parent;
        std::vector<Node *> children;
        Node(const Board &state, int depth, Node* parent = nullptr): 
            state(state), value(0.0), depth(depth), is_leaf(false), parent(parent) {}
        ~Node() {
            for (auto child : children) {
                delete child;
            }
        }

        virtual void expand() = 0;
        virtual void pass_value_up(const NTupleTD &agent) = 0;
};

class MaxNode: public Node
{
    public:
        MaxNode(const Board &state, int depth, Node *parent = nullptr): 
            Node(state, depth, parent) {}
        void expand() override;
        void pass_value_up(const NTupleTD &agent) override;
};

class ChanceNode: public Node
{
    public:
        double reward;      // Reward obtained to reach this node
        ChanceNode(const Board &state, int depth, double reward, Node *parent = nullptr): 
            reward(reward), Node(state, depth, parent) {}
        void expand() override;
        void pass_value_up(const NTupleTD &agent) override;
};

std::queue<Node *> expand_queue;
std::queue<Node *> pass_value_queue;

int Expectimax(const Board &root, const NTupleTD &agent, int depth, int num_sample = DEFAULT_NUM_SAMPLE);

#endif