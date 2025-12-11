#include <cstdlib>
#include <cmath>
#include <limits>
#include <random>
#include <iostream>
#include <list>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "mcts.hpp"
#include "../env/2048env.hpp"
#include "../TD_learning_sequential_ver/n_tuple_TD.hpp"

/* Forward declaration */
struct MCTSNode;
struct DecisionNode;
struct ChanceNode;

struct Task {
    bool is_chance;  // True if is root is chance node
    bool cancel = false;
    MCTSNode *futureRoot = nullptr;
    Task (MCTSNode *futureRoot, bool is_chance)
        : is_chance(is_chance), futureRoot(futureRoot) {}
};

class ThreadPool {
private:
    std::vector<Env2048> envs;
    std::vector<std::thread> pool;
    std::list<std::shared_ptr<Task>> tasks;
    std::mutex queue_mutex;
    std::condition_variable queue_cv;
    bool stop = false;

public:
    ThreadPool(const unsigned int num_threads);
    ~ThreadPool();
    void enqueue(std::shared_ptr<Task> task);
};

struct Stats {
    int cumulate_score = 0;
    int visit_count = 0;
    double total_reward = 0.0;
    double min_avg = std::numeric_limits<double>::infinity();
    double max_avg = -std::numeric_limits<double>::infinity();
    
    double avg_reward() const {
        return this->total_reward / this->visit_count;
    }
    void update_reward(const double reward, double &min_avg, double &max_avg);
    Stats(const int cumulate_score): cumulate_score(cumulate_score) {}
};

struct FutureProp {
    bool future;  // True if this node is for future, i.e. not on the sequential tree
    bool working;  // False if (fully_expanded && next_step == nullptr)
    std::atomic<bool> worker_processing;  // True if worker is working on this task
    std::atomic<bool> worker_finished;  // True if fully expanded
    int cur_reserve = 0;  // Remove from queue if cur_reserve >= max_reserve
    int max_reserve;  // Useless for decision node, since only 4 children
    double reward = 0.0;  // Store rollout reward for future node
    Stats stats;  // A copy for worker to maintain scores in future

    DecisionNode *next_step = nullptr;  // Points to next expanded future node
    std::shared_ptr<Task> pending_task = nullptr;  // Points to task in queue, for main to cancel

    // Protect cur_reserve, next_step, pending_task
    std::mutex mutex;  // Since both main and worker will access

    FutureProp(const bool working, const int max_reserve, const int cumulate_score)
        : working(working), max_reserve(max_reserve), stats(cumulate_score) {}
};


void Stats::update_reward(const double reward, double &min_avg, double &max_avg)
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

struct MCTSNode {
public:
    Board board;
    Stats stats;
    FutureProp fprop;

    virtual MCTSNode *get_parent() { return nullptr; }
    virtual bool fully_expanded_future() { return false; }

protected:
    MCTSNode(const Board &board, const int cumulate_score, const bool future)
        : stats(cumulate_score) {
        this->board = board;
    }
};

struct DecisionNode : public MCTSNode {
    bool game_over;
    std::vector<int> untried_actions;
    ChanceNode *parent;
    std::vector<ChanceNode *> children;

    DecisionNode(Env2048 &env, ChanceNode *parent, const Board &board,
                 const int cumulate_score, const bool future)
        : MCTSNode(board, cumulate_score, future), parent(parent) {
        env.set_board(this->board);
        this->untried_actions = env.get_legal_actions();
        this->game_over = this->untried_actions.empty();
    }
    ChanceNode *select_child(const double explore_c, const bool is_worker) const;
    DecisionNode *expand_child(Env2048 &env, const int action_idx);
    MCTSNode *get_parent() override final;
    bool fully_expanded_future() override final;
};

struct ChanceNode : public MCTSNode {
    int action;
    int max_children = 0;
    DecisionNode *parent;
    std::vector<DecisionNode *> children;

    ChanceNode(DecisionNode *parent, const Board &board, const int action,
               const int cumulate_score, const bool future, const bool working,
               const int max_future_step);
    DecisionNode *select_child(Env2048 &env, bool &expanded, const bool is_worker);
    DecisionNode *expand_child(Env2048 &env, const Board &board);
    double uct_value(const double explore_c, const bool is_worker) const;
    MCTSNode *get_parent() override final;
    bool fully_expanded_future() override final;
};

ChanceNode::ChanceNode(DecisionNode *parent, const Board &board, const int action,
                       const int cumulate_score, const bool future, const bool working,
                       const int max_future_step)
    : MCTSNode(board, cumulate_score, future), action(action), parent(parent)
{
    this->max_children = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == 0)
                this->max_children += 2;
        }
    }
}

double ChanceNode::uct_value(const double explore_c, const bool is_worker) const
{
    if (!is_worker) {
        return this->stats.avg_reward()
            + (this->parent->stats.max_avg - this->parent->stats.min_avg) * explore_c
            * sqrt(log(this->parent->stats.visit_count) / this->stats.visit_count);
    }
    return this->fprop.stats.avg_reward()
        + (this->parent->fprop.stats.max_avg - this->parent->fprop.stats.min_avg) * explore_c
        * sqrt(log(this->parent->fprop.stats.visit_count) / this->fprop.stats.visit_count);
}

MCTSNode *DecisionNode::get_parent()
{
    return this->parent;
}

MCTSNode *ChanceNode::get_parent()
{
    return this->parent;
}

bool DecisionNode::fully_expanded_future()
{
    return (this->untried_actions.size() == 0);
}

bool ChanceNode::fully_expanded_future()
{
    return (this->children.size() == this->max_children);
}

ChanceNode *DecisionNode::select_child(const double explore_c, const bool is_worker) const
{
    double best_uct = -std::numeric_limits<double>::infinity();
    ChanceNode *best_child = nullptr;
    for (ChanceNode *child : this->children) {
        const double uct = child->uct_value(explore_c, is_worker);
        if (uct > best_uct) {
            best_uct = uct;
            best_child = child;
        }
    }
    return best_child;
}

DecisionNode *ChanceNode::select_child(Env2048 &env, bool &expanded, const bool is_worker)
{
    env.set_board(this->board);
    env.add_random_tile();
    for (DecisionNode *child : this->children) {
        if (env.get_board() == child->board) {
            expanded = false;
            return child;
        }
    }
    // At a new state, only worker should be here
    if (!is_worker) {
        std::cerr << "[BAD] In ChanceNode::select_child(), non-working node should have all child expanded." << std::endl;
        exit(1);
    }
    expanded = true;
    // TODO expand child for worker
    return this->expand_child(env, env.get_board());
}

// Only called by worker
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

// Only called by worker
DecisionNode *ChanceNode::expand_child(Env2048 &env, const Board &board)
{
    DecisionNode *child = new DecisionNode(env, this, board, this->cumulate_score);
    this->children.push_back(child);
    return child;
}

class MCTS {
public:
    MCTS(const Board &board, const NTupleTD &agent, const double explore_c, const int rollout_depth);
    ~MCTS();
    void run_main();
    void run_worker(Env2048 &env, MCTSNode *futureRoot, const bool is_chance);
    int get_best_action() const;

private:
    DecisionNode root;
    const NTupleTD &agent;
    ThreadPool pool;
    const double explore_c;
    const int rollout_depth;
    std::mt19937 rng;

    MCTS();
    DecisionNode *select_and_expand(DecisionNode *root);
    bool stop_working_main(MCTSNode *futureRoot);
    void enqueue_task_main(MCTSNode *futureRoot, const bool is_chance);
    DecisionNode *get_next_main(MCTSNode *futureRoot, const bool is_chance);
    DecisionNode *select_and_expand_main();
    DecisionNode *expand_workerD(Env2048 &env, DecisionNode *root);
    DecisionNode *expand_workerC(Env2048 &env, ChanceNode *root);
    double rollout(Env2048 &env, const DecisionNode *leaf) const;
    void backpropagate(DecisionNode *leaf, double reward);
    void backpropagate_main(DecisionNode *leaf, double reward) const;
    void backpropagate_worker(MCTSNode *futureRoot, DecisionNode *leaf, double reward);
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

// Needs to acquire mutex before caling
bool MCTS::stop_working_main(MCTSNode *futureRoot)
{
    if (futureRoot->fprop.cur_reserve > 0
    || futureRoot->fprop.worker_finished.load() == false)
        return false;

    // futureRoot finished working
    if (futureRoot->fprop.next_step != nullptr) {
        std::cerr << "[BAD] In stop_working(), next_step should be null" << std::endl;
        exit(1);
    }
    if (futureRoot->fprop.pending_task != nullptr) {
        futureRoot->fprop.pending_task->cancel = true;
    }
    futureRoot->fprop.working = false;
    return true;
}

void MCTS::enqueue_task_main(MCTSNode *futureRoot, const bool is_chance)
{
    if (futureRoot->fprop.worker_finished.load() == true)
        return;
    std::shared_ptr<Task> task = std::make_shared<Task>(futureRoot, is_chance);
    std::unique_lock<std::mutex> lock(futureRoot->fprop.mutex);
    futureRoot->fprop.pending_task = task;
    this->pool.enqueue(task);
}

DecisionNode *MCTS::get_next_main(MCTSNode *futureRoot, const bool is_chance)
{
    std::unique_lock<std::mutex> lock(futureRoot->fprop.mutex);
    if (futureRoot->fprop.next_step != nullptr) {
        // Use next step
        DecisionNode *next = futureRoot->fprop.next_step;
        futureRoot->fprop.next_step = next->fprop.next_step;
        next->fprop.next_step = nullptr;
        futureRoot->fprop.cur_reserve--;
        if (this->stop_working_main(futureRoot)
        || futureRoot->fprop.pending_task != nullptr
        || futureRoot->fprop.worker_processing.load() == true)
            return next;
        lock.unlock();
        // Work not finished and pending and worker not processing, enqueue
        this->enqueue_task_main(futureRoot, is_chance);
        return next;
    }
    if (futureRoot->fprop.pending_task == nullptr) {
        std::cerr << "[BAD] In MCTS::get_next_step, next is null and worker is working on next, but worker did not lock" << std::endl;
        exit(1);
    }
    // No next step and pending in queue
    futureRoot->fprop.pending_task->cancel = true;
    lock.unlock();
    std::cerr << "[LOG] Degraded to sequential expand by main thread" << std::endl;
    DecisionNode *ret = (is_chance)? this->expand_workerC(static_cast<ChanceNode *>(futureRoot))
                                   : this->expand_workerD(static_cast<DecisionNode *>(futureRoot));
    lock.lock();
    if (this->stop_working_main(futureRoot))
        return ret;
    lock.unlock();
    this->enqueue_task_main(futureRoot, is_chance);
    return ret;
}

DecisionNode *MCTS::select_and_expand_main()
{
    DecisionNode *cursorD = &this->root;
    // TODO lock working?
    while (!cursorD->game_over && !cursorD->fprop.working) {
        ChanceNode *cursorC = cursorD->select_child(this->explore_c, false);
        if (cursorC->fprop.working) {
            cursorD = this->get_next_main(cursorC, true);
            if (cursorD->fprop.working == false) {
                // TODO add to task queue if futureRoot fully expanded
            }
            return cursorD;
        }
        bool expanded;  // Dummy, useless for main thread
        cursorD = cursorC->select_child(this->env, expanded, false);
    }
    if (cursorD->game_over)
        return cursorD;

    // Do cursorD need to expand all child by worker? yes
    // cursorD is working
    // TODO return next_step or expand if pending
    return;
}

DecisionNode *MCTS::expand_workerD(Env2048 &env, DecisionNode *root)
{
    // Root is where to expand, since stop working once fully expanded
    if (root->untried_actions.size() == 0) {
        std::cerr << "[BAD] select_and_expand_workerD only accept non-fully expanded root" << std::endl;
        exit(1);
    }
    if (root->game_over)
        return root;
    std::uniform_int_distribution<> dis(0, root->untried_actions.size() - 1);
    const int action_idx = dis(this->rng);
    // TODO set root worker finished
    return root->expand_child(env, action_idx);
}

DecisionNode *MCTS::expand_workerC(Env2048 &env, ChanceNode *root)
{
    // Starting from root, may need to select down the tree if select result in a old child
    bool expanded = false;
    DecisionNode *cursorD = root->select_child(env, expanded, true);
    while (!cursorD->game_over && !expanded && cursorD->untried_actions.size() == 0) {
        ChanceNode *cursorC = cursorD->select_child(this->explore_c, true);
        expanded = false;
        cursorD = cursorC->select_child(env, expanded, true);
    }
    if (cursorD->game_over || expanded)
        return cursorD;
    // Expand at decision node
    std::uniform_int_distribution<> dis(0, cursorD->untried_actions.size() - 1);
    const int action_idx = dis(this->rng);
    // TODO set root worker finished
    return cursorD->expand_child(env, action_idx);
}

// Only called by worker
double MCTS::rollout(Env2048 &env, const DecisionNode *leaf) const
{
    env.set_board(leaf->board);
    env.set_score(0);
    Board after_state = leaf->board;
    bool game_over = env.is_game_over();
    for (int round = 0; !game_over && round < this->rollout_depth; round++) {
        std::vector<int> legal_actions = env.get_legal_actions();
        std::uniform_int_distribution<> dis(0, legal_actions.size() - 1);
        StepResult result = env.step(legal_actions[dis(this->rng)]);
        after_state = result.board;
        game_over = result.game_over;
    }
    if (game_over)
        return leaf->fprop.stats.cumulate_score + env.get_score();
    return leaf->fprop.stats.cumulate_score + env.get_score() + this->agent.cal_value(after_state);
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

void MCTS::backpropagate_main(DecisionNode *leaf, double reward) const
{
    MCTSNode *cursor = leaf;
    double min_avg = std::numeric_limits<double>::infinity(),
           max_avg = -std::numeric_limits<double>::infinity();
    while (cursor != nullptr) {
        cursor->stats.update_reward(reward, min_avg, max_avg);
        cursor = cursor->get_parent();
    }
}

void MCTS::backpropagate_worker(MCTSNode *futureRoot, DecisionNode *leaf, double reward)
{
    MCTSNode *cursor = leaf;
    double min_avg = std::numeric_limits<double>::infinity(),
           max_avg = -std::numeric_limits<double>::infinity();
    // Only back propagate until future root
    while (cursor != nullptr && cursor != futureRoot) {
        cursor->fprop.stats.update_reward(reward, min_avg, max_avg);
        cursor = cursor->get_parent();
    }
}

void MCTS::run_main()
{
    DecisionNode *leaf = this->select_and_expand_main();
    // TODO lock?
    leaf->fprop.future = false;
    this->backpropagate_main(leaf, leaf->fprop.reward);
}

void MCTS::run_worker(Env2048 &env, MCTSNode *futureRoot, const bool is_chance)
{
    // futureRoot working should be true
    // TODO lock?
    while (futureRoot->fprop.cur_reserve < futureRoot->fprop.max_reserve) {
        DecisionNode *leaf = nullptr;
        if (is_chance) {
            leaf = this->expand_workerC(static_cast<ChanceNode *>(futureRoot));
        } else {
            leaf = this->expand_workerD(static_cast<DecisionNode *>(futureRoot));
        }
        leaf->fprop.reward = this->rollout(env, leaf);
        this->backpropagate_worker(futureRoot, leaf, leaf->fprop.reward);
        
        leaf->fprop.cur_reserve++;
    }
}

int MCTS::get_best_action() const
{
    int most_visit = -1;
    int action = -1;
    for (ChanceNode *child : this->root.children) {
        if (child->stats.visit_count > most_visit) {
            most_visit = child->stats.visit_count;
            action = child->action;
        }
    }
    return action;
}

ThreadPool::ThreadPool(const unsigned int num_threads)
{
    this->envs.resize(num_threads);
    for (unsigned int i = 0; i < num_threads; i++) {
        this->pool.emplace_back([this, i] {
            while (true) {
                bool assigned = false;
                std::shared_ptr<Task> task;
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
                    workerFunction(this->envs[i], task->futureRoot);
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

void ThreadPool::enqueue(std::shared_ptr<Task> task)
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
        mcts.run_main();
    }
    return mcts.get_best_action();
}
