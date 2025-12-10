#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include "mcts.hpp"
#include "../env/2048env.hpp"
#include "../TD_learning_sequential_ver/n_tuple_TD.hpp"

/* Forward declaration */
struct DecisionNode;
struct ChanceNode;

struct MCTSNode {
public:
    Board board;
    int cumulate_score = 0;
    int visit_count = 0;
    double total_reward = 0.0;
    double min_avg = std::numeric_limits<double>::infinity();
    double max_avg = -std::numeric_limits<double>::infinity();

    double avg_reward() const {
        return this->total_reward / this->visit_count;
    }
    void update_reward(const double reward, double &min_avg, double &max_avg);
    virtual MCTSNode *get_parent() { return nullptr; }

protected:
    MCTSNode(const Board &board, const int cumulate_score)
        : cumulate_score(cumulate_score) {
        this->board = board;
    }
};

struct DecisionNode : public MCTSNode {
    bool game_over;
    std::vector<int> untried_actions;
    ChanceNode *parent;
    std::vector<ChanceNode *> children;

    DecisionNode(Env2048 &env, ChanceNode *parent, const Board &board, const int cumulate_score)
        : MCTSNode(board, cumulate_score), parent(parent) {
        env.set_board(this->board);
        this->untried_actions = env.get_legal_actions();
        this->game_over = this->untried_actions.empty();
    }
    ChanceNode *select_child(const double explore_c);
    DecisionNode *expand_child(Env2048 &env, const int action_idx);
    MCTSNode *get_parent() override final;
};

struct ChanceNode : public MCTSNode {
    int action;
    DecisionNode *parent;
    std::vector<DecisionNode *> children;

    ChanceNode(DecisionNode *parent, const Board &board, const int action, const int cumulate_score)
        : MCTSNode(board, cumulate_score), action(action), parent(parent) {}
    DecisionNode *select_child(Env2048 &env, bool &expanded);
    DecisionNode *expand_child(Env2048 &env, const Board &board);
    double uct_value(const double explore_c) const {
        // std::cout << "[UCT] "<< this->avg_reward() << " <-> " << explore_c * sqrt(log(this->parent->visit_count) / this->visit_count) << std::endl;
        return this->avg_reward() + (this->parent->max_avg - this->parent->min_avg) * explore_c
                                  * sqrt(log(this->parent->visit_count) / this->visit_count);
    }
    MCTSNode *get_parent() override final;
};

MCTSNode *DecisionNode::get_parent()
{
    return this->parent;
}

MCTSNode *ChanceNode::get_parent()
{
    return this->parent;
}

void MCTSNode::update_reward(const double reward, double &min_avg, double &max_avg)
{
    this->visit_count += 1;
    this->total_reward += reward;
    double avg = this->avg_reward();
    min_avg = std::min(min_avg, avg);
    min_avg = std::min(min_avg, this->min_avg);
    this->min_avg = min_avg;
    max_avg = std::max(max_avg, avg);
    max_avg = std::max(max_avg, this->max_avg);
    this->max_avg = max_avg;
}

ChanceNode *DecisionNode::select_child(const double explore_c)
{
    double best_uct = -std::numeric_limits<double>::infinity();
    ChanceNode *best_child = nullptr;
    for (ChanceNode *child : this->children) {
        const double uct = child->uct_value(explore_c);
        if (uct > best_uct) {
            best_uct = uct;
            best_child = child;
        }
    }
    return best_child;
}

DecisionNode *ChanceNode::select_child(Env2048 &env, bool &expanded)
{
    // TODO Progressive Widening?
    env.set_board(this->board);
    env.add_random_tile();
    for (DecisionNode *child : this->children) {
        if (env.get_board() == child->board) {
            expanded = false;
            return child;
        }
    }
    // At a new state
    expanded = true;
    return this->expand_child(env, env.get_board());
}

DecisionNode *DecisionNode::expand_child(Env2048 &env, const int action_idx)
{
    const int action = this->untried_actions[action_idx];
    this->untried_actions.erase(this->untried_actions.begin() + action_idx);
    env.set_board(this->board);
    env.set_score(0);
    StepResult result = env.step(action);
    ChanceNode *child = new ChanceNode(this, result.board, action, result.score + this->cumulate_score);
    this->children.push_back(child);
    return child->expand_child(env, env.get_board());
}

DecisionNode *ChanceNode::expand_child(Env2048 &env, const Board &board)
{
    DecisionNode *child = new DecisionNode(env, this, board, this->cumulate_score);
    this->children.push_back(child);
    return child;
}

class MCTS {
public:
    Env2048 env;
    MCTS(const Board &board, const NTupleTD &agent, const double explore_c, const int rollout_depth);
    ~MCTS();
    void run();
    int get_best_action() const;

private:
    DecisionNode root;
    const NTupleTD &agent;
    const double explore_c;
    const int rollout_depth;
    std::mt19937 rng;

    MCTS();
    DecisionNode *select_and_expand(DecisionNode *root);
    double rollout(DecisionNode *leaf);
    void backpropagate(DecisionNode *leaf, double reward);
    void delete_tree(DecisionNode *node);
};

MCTS::MCTS(const Board &board, const NTupleTD &agent,
           const double explore_c, const int rollout_depth)
    : root(this->env, nullptr, board, 0), agent(agent), explore_c(explore_c),
      rollout_depth(rollout_depth)
{
    this->rng.seed(1);
}

void MCTS::delete_tree(DecisionNode *node)
{
    for (ChanceNode *cursorC : node->children) {
        for (DecisionNode *cursorD : cursorC->children) {
            delete_tree(cursorD);
            delete cursorD;
        }
        delete cursorC;
    }
}

MCTS::~MCTS()
{
    this->delete_tree(&this->root);
}

DecisionNode *MCTS::select_and_expand(DecisionNode *root)
{
    DecisionNode *cursorD = root;
    while (!cursorD->game_over && cursorD->untried_actions.size() == 0) {
        ChanceNode *cursorC = cursorD->select_child(this->explore_c);
        bool expanded = false;
        cursorD = cursorC->select_child(this->env, expanded);
        if (expanded)
            return cursorD;
    }
    if (cursorD->game_over)
        return cursorD;
    // Expand at decision node
    std::uniform_int_distribution<> dis(0, cursorD->untried_actions.size() - 1);
    const int action_idx = dis(this->rng);
    return cursorD->expand_child(this->env, action_idx);
}

double MCTS::rollout(DecisionNode *leaf)
{
    this->env.set_board(leaf->board);
    this->env.set_score(0);
    Board after_state = leaf->board;
    bool game_over = this->env.is_game_over();
    for (int round = 0; round < this->rollout_depth; round++) {
        if (game_over)
            return leaf->cumulate_score + this->env.get_score();
        std::vector<int> legal_actions = env.get_legal_actions();
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        StepResult result = env.step(legal_actions[dis(this->rng)]);
        after_state = result.board;
        game_over = result.game_over;
    }
    if (game_over)
        return leaf->cumulate_score + this->env.get_score();
    // std::cout << "[REWARD]" << leaf->cumulate_score + this->env.get_score() << " <-> " << this->agent.cal_value(this->env.get_board()) << std::endl;
    
    return leaf->cumulate_score + this->env.get_score() + this->agent.cal_value(after_state);
}

void MCTS::backpropagate(DecisionNode *leaf, double reward)
{
    MCTSNode *cursor = leaf;
    double min_avg = std::numeric_limits<double>::infinity(),
           max_avg = -std::numeric_limits<double>::infinity();
    while (cursor != nullptr) {
        cursor->update_reward(reward, min_avg, max_avg);
        cursor = cursor->get_parent();
    }
}

void MCTS::run()
{
    DecisionNode *expanded = this->select_and_expand(&this->root);
    double reward = this->rollout(expanded);
    this->backpropagate(expanded, reward);
}

int MCTS::get_best_action() const
{
    int most_visit = -1;
    int action = -1;
    for (ChanceNode *child : this->root.children) {
        if (child->visit_count > most_visit) {
            most_visit = child->visit_count;
            action = child->action;
        }
    }
    return action;
}

class ThreadPool {
private:
    std::vector<std::thread> pool;
    std::list<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool stop = false;

public:
    ThreadPool(const unsigned int num_threads);
    ~ThreadPool();
    void enqueue(std::function<void()> task);
};

ThreadPool::ThreadPool(const unsigned int num_threads)
{
    for (unsigned int i = 0; i < num_threads; i++) {
        this->pool.emplace_back([this] {
            while (true) {
                bool assigned = false;
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
                    this->queue_cv.wait(lock, [this] {
                        return (!this->tasks.empty() || this->stop);
                    });
                    if (this->stop)
                        return;
                    if (!this->tasks.empty()) {
                        task = std::move(this->tasks.front());
                        this->tasks.pop_front();
                        assigned = true;
                    }
                }
                if (assigned)
                    task();
            }
        });
    }
}

ThreadPool::~ThreadPool()
{
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        this->stop = true;
    }
    this->queue_cv.notify_all();
    for (std::thread &worker : this->pool) {
        worker.join();
    }
}

void ThreadPool::enqueue(std::function<void()> task)
{
    {
        std::unique_lock<std::mutex> lock(this->queue_mutex);
        this->tasks.emplace_back(std::move(task));
    }
    this->queue_cv.notify_one();
}

int mcts_action(const Board &board, const NTupleTD &agent,
                const double exploration_constant,
                const int iterations, const int rollout_depth)
{
    MCTS mcts(board, agent, exploration_constant, rollout_depth);
    for (int it = 0; it < iterations; it++) {
        mcts.run();
    }
    return mcts.get_best_action();
}
