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

// TODO (1) Why fully expanded?
//      (2) Why cur_reserve not match?

// #define DEBUG
// #define DEBUG2

#define TID (std::hash<std::thread::id>{}(std::this_thread::get_id()) % 1000)

/* Forward declaration */
struct MCTSNode;
struct DecisionNode;
struct ChanceNode;
class MCTS;

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
    ThreadPool(MCTS &mcts, const unsigned int num_threads);
    void enqueue(std::shared_ptr<Task> task);
    void stop_all();
};

struct Stats {
    int cumulate_score = 0;  // fprop.stats.cumulate_score is useless
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

    FutureProp(const bool future, const bool working, const int max_reserve)
        : future(future), working(working), max_reserve(max_reserve), stats(0)
    {
        this->worker_processing.store(false);
        this->worker_finished.store(false);
    }

    bool remain_work() {
        std::unique_lock<std::mutex> lock(this->mutex);
        if (this->cur_reserve < this->max_reserve) {
            return true;
        }
        return false;
    }
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

#ifdef DEBUG2
std::atomic<int> GLOBAL_ID = 0;
#endif

struct MCTSNode {
public:
    Board board;
    Stats stats;
    FutureProp fprop;

#ifdef DEBUG2
    int id;
#endif

    virtual MCTSNode *get_parent() { return nullptr; }
    virtual bool fully_expanded_future() { return false; }
    virtual bool all_child_non_future() { return false; }

protected:
    MCTSNode(const Board &board, const int cumulate_score,
             const bool future, const bool working, const int max_reserve)
        : stats(cumulate_score), fprop(future, working, max_reserve) {
        this->board = board;
#ifdef DEBUG2
        this->id = GLOBAL_ID.fetch_add(1);
        std::cerr << "[ID] " << this->id  << " created" << std::endl;
#endif
    }
};

struct DecisionNode : public MCTSNode {
    bool game_over;
    std::vector<int> untried_actions;
    ChanceNode *parent;
    std::vector<ChanceNode *> children;

    DecisionNode(Env2048 &env, ChanceNode *parent, const Board &board,
                 const int cumulate_score, const bool future, const bool working);
    ChanceNode *select_child(const double explore_c, const bool is_worker) const;
    DecisionNode *expand_child_worker(Env2048 &env, const int action_idx);
    MCTSNode *get_parent() override final;
    bool fully_expanded_future() override final;
    bool all_child_non_future() override final;
};

struct ChanceNode : public MCTSNode {
    int action;
    int max_children = 0;
    DecisionNode *parent;
    std::vector<DecisionNode *> children;

    ChanceNode(DecisionNode *parent, const Board &board, const int action,
               const int cumulate_score, const bool future, const bool working,
               const int max_reserve);
    DecisionNode *select_child(Env2048 &env, bool &expanded, const bool is_worker);
    DecisionNode *expand_child_worker(Env2048 &env, const Board &board);
    double uct_value(const double explore_c, const bool is_worker) const;
    MCTSNode *get_parent() override final;
    bool fully_expanded_future() override final;
    bool all_child_non_future() override final;
};

DecisionNode::DecisionNode(Env2048 &env, ChanceNode *parent, const Board &board,
                           const int cumulate_score, const bool future, const bool working)
    : MCTSNode(board, cumulate_score, future, working, 0), parent(parent)
{
    env.set_board(this->board);
    this->untried_actions = env.get_legal_actions();
    this->game_over = this->untried_actions.empty();
    // TODO Tune max reserve, decrease as deeper in tree
    this->fprop.max_reserve = this->untried_actions.size();
#ifdef DEBUG2
    if (this->parent != nullptr)
        std::cerr << "[ID] " << this->id  << " parent " << this->parent->id << std::endl;
#endif
}

ChanceNode::ChanceNode(DecisionNode *parent, const Board &board, const int action,
                       const int cumulate_score, const bool future, const bool working,
                       const int max_reserve)
    : MCTSNode(board, cumulate_score, future, working, max_reserve), action(action), parent(parent)
{
    this->max_children = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (board[i][j] == 0)
                this->max_children += 2;
        }
    }
    this->fprop.max_reserve = std::min(fprop.max_reserve, this->max_children);
#ifdef DEBUG2
    if (this->parent != nullptr)
        std::cerr << "[ID] " << this->id  << " parent " << this->parent->id << std::endl;
#endif
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

bool DecisionNode::all_child_non_future()
{
    if (this->fprop.future)
        return false;
    for (ChanceNode *child : this->children) {
        if (child->fprop.future)
            return false;
    }
    return true;
}

bool ChanceNode::all_child_non_future()
{
    if (this->fprop.future)
        return false;
    for (DecisionNode *child : this->children) {
        if (child->fprop.future)
            return false;
    }
    return true;
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
    return this->expand_child_worker(env, env.get_board());
}

// Only called by worker
DecisionNode *DecisionNode::expand_child_worker(Env2048 &env, const int action_idx)
{
    const int action = this->untried_actions[action_idx];
    this->untried_actions.erase(this->untried_actions.begin() + action_idx);
    env.set_board(this->board);
    env.set_score(0);
    StepResult result = env.step(action);
    // TODO Tune max reserve, decrease as deeper in tree
    ChanceNode *child = new ChanceNode(this, result.board, action,
                                       result.score + this->stats.cumulate_score,
                                       true, false, 10);
    this->children.push_back(child);
    return child->expand_child_worker(env, env.get_board());
}

// Only called by worker
DecisionNode *ChanceNode::expand_child_worker(Env2048 &env, const Board &board)
{
    DecisionNode *child = new DecisionNode(env, this, board,
                                           this->stats.cumulate_score,
                                           true, false);
    this->children.push_back(child);
    return child;
}

class MCTS {
public:
    MCTS(const Board &board, const NTupleTD &agent, const unsigned int num_threads,
         const double explore_c, const int rollout_depth);
    ~MCTS();
    void run_main();
    void run_worker(Env2048 &env, std::shared_ptr<Task> task);
    int get_best_action() const;
    void terminate() {
        this->pool.stop_all();
    }

private:
    Env2048 main_env;
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
    void post_next_main(MCTSNode *futureRoot, const bool is_chance);
    DecisionNode *select_and_expand_main(MCTSNode * &futureRoot);
    DecisionNode *expand_workerD(Env2048 &env, DecisionNode *root);
    DecisionNode *expand_workerC(Env2048 &env, ChanceNode *root);
    double rollout_worker(Env2048 &env, const DecisionNode *leaf);
    void backpropagate(DecisionNode *leaf, double reward);
    void backpropagate_main(DecisionNode *leaf, double reward) const;
    void backpropagate_worker(MCTSNode *futureRoot, DecisionNode *leaf, double reward);
    void delete_tree(DecisionNode *node);
};

MCTS::MCTS(const Board &board, const NTupleTD &agent, const unsigned int num_threads,
           const double explore_c, const int rollout_depth)
    : root(this->main_env, nullptr, board, 0, false, true),
      agent(agent), pool(*this, num_threads),
      explore_c(explore_c), rollout_depth(rollout_depth)
{
    this->rng.seed(1);
    this->enqueue_task_main(&this->root, false);
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
    if (futureRoot->fprop.pending_task != nullptr)
        futureRoot->fprop.pending_task->cancel = true;
    futureRoot->fprop.pending_task = task;
    this->pool.enqueue(task);
}

/* 
 * If return is not from next, than future is false.
 * futureRoot should be working
 */
DecisionNode *MCTS::get_next_main(MCTSNode *futureRoot, const bool is_chance)
{
#ifdef DEBUG
    std::cerr << "[MAIN] In get next" << std::endl;
#endif
    // Spin lock, change to cv?
    while (true) {
#ifdef DEBUG
        std::cerr << "[MAIN] Before lock" << std::endl;
#endif
        std::unique_lock<std::mutex> lock(futureRoot->fprop.mutex);
#ifdef DEBUG
        std::cerr << "[MAIN] After lock" << std::endl;
#endif
        if (futureRoot->fprop.next_step != nullptr) {
            // Use next step
#ifdef DEBUG
            std::cerr << "Main use next step" << std::endl;
#endif
            DecisionNode *next = futureRoot->fprop.next_step;
            futureRoot->fprop.next_step = next->fprop.next_step;
            next->fprop.next_step = nullptr;
            next->fprop.future = true;
            futureRoot->fprop.cur_reserve--;
#ifdef DEBUG2
            assert(futureRoot != nullptr);
            std::cerr << "[ID] " << futureRoot->id
                << " use next step " << next->id
                << ", reserve: " << futureRoot->fprop.cur_reserve << std::endl;
            if (futureRoot->fprop.next_step != nullptr) {
                std::cerr << "[ID] " << futureRoot->id
                    << " new next step" << futureRoot->fprop.next_step->id << std::endl;
            }
#endif
            if (this->stop_working_main(futureRoot)
            || futureRoot->fprop.pending_task != nullptr
            || futureRoot->fprop.worker_processing.load() == true)
                return next;
            lock.unlock();
            // Work not finished and not pending and worker not processing, enqueue
            this->enqueue_task_main(futureRoot, is_chance);
            return next;
        }
        if (futureRoot->fprop.worker_processing.load()) {
#ifdef DEBUG
            std::cerr << "Spinning worker propcessing" << std::endl;
#endif
            lock.unlock();
            // std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }
        if (futureRoot->fprop.pending_task == nullptr) {
            // TODO cv?
            // next is null, and worker is working on it, spin until worker finished
            if (futureRoot->fully_expanded_future()) {
                futureRoot->fprop.worker_finished.store(true);
                futureRoot->fprop.working = false;
                return nullptr;
            }
            std::cerr << "Hi" << std::endl;
            exit(1);
            lock.unlock();
            continue;
        }
        // No next step and pending in queue
        // TODO uncomment
        futureRoot->fprop.pending_task->cancel = true;
        lock.unlock();
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
        break;
    }
#ifdef DEBUG
    std::cerr << "[LOG] Degraded to sequential expand by main thread" << std::endl;
#endif
    DecisionNode *ret = (is_chance)? this->expand_workerC(this->main_env, static_cast<ChanceNode *>(futureRoot))
                                   : this->expand_workerD(this->main_env, static_cast<DecisionNode *>(futureRoot));
    if (ret != nullptr)
        ret->fprop.future = false;
    std::unique_lock<std::mutex> lock(futureRoot->fprop.mutex);
    if (futureRoot->fully_expanded_future())
        futureRoot->fprop.worker_finished.store(true);
    if (this->stop_working_main(futureRoot))
        return ret;
    lock.unlock();
    this->enqueue_task_main(futureRoot, is_chance);
    return ret;
}

void MCTS::post_next_main(MCTSNode *futureRoot, const bool is_chance)
{
#ifdef DEBUG
    std::cerr << "In post next main" << std::endl;
#endif
    // if (!futureRoot->fprop.worker_finished.load()) {
    //     return;
    // }
    if (futureRoot->fprop.working)
        return;
    // Set working for child that
    // (1) Non-fully_expanded child (add to queue)
    // or
    // (2) fully_expanded but some child is future
    // If child is not working, than recurse for child
    if (is_chance) {
        for (DecisionNode *child : static_cast<ChanceNode *>(futureRoot)->children) {
            const bool fully_expanded = child->fully_expanded_future();
            if (fully_expanded)
                child->fprop.worker_finished.store(true);
            bool temp = child->all_child_non_future();
            child->fprop.working = !(fully_expanded && temp);
            if (!fully_expanded) {
                this->enqueue_task_main(child, !is_chance);
            } else if (!child->fprop.working) {
                this->post_next_main(child, !is_chance);
            }
        }
    } else {
        for (ChanceNode *child : static_cast<DecisionNode *>(futureRoot)->children) {
            const bool fully_expanded = child->fully_expanded_future();
            if (fully_expanded)
                child->fprop.worker_finished.store(true);
            bool temp = child->all_child_non_future();
            child->fprop.working = !(fully_expanded && temp);
            if (!fully_expanded) {
                this->enqueue_task_main(child, !is_chance);
            } else if (!child->fprop.working) {
                this->post_next_main(child, !is_chance);
            }
        }
    }
}

DecisionNode *MCTS::select_and_expand_main(MCTSNode * &futureRoot)
{
    DecisionNode *cursorD = &this->root;
    while (!cursorD->game_over && !cursorD->fprop.working) {
        ChanceNode *cursorC = cursorD->select_child(this->explore_c, false);
        if (cursorC == nullptr) {
            // std::cerr << "[BAD] select null" << std::endl;
            return nullptr;
        }
        if (cursorC->fprop.working) {
            cursorD = this->get_next_main(cursorC, true);
            this->post_next_main(cursorC, true);
            if (cursorD != nullptr) {
                futureRoot = cursorC;
                return cursorD;
            }
        }
        bool expanded;  // Dummy, useless for main thread
        cursorD = cursorC->select_child(this->main_env, expanded, false);
    }
    futureRoot = cursorD;
    if (cursorD->game_over)
        return cursorD;

    // Does cursorD need to expand all child by worker? yes
    // cursorD is working
    DecisionNode *ret = this->get_next_main(cursorD, false);
    this->post_next_main(cursorD, false);
    return ret;
}

DecisionNode *MCTS::expand_workerD(Env2048 &env, DecisionNode *root)
{
#ifdef DEBUG2
    std::cerr << "[DEBUG] " << root->id << " In expand workerD" << std::endl;
#endif
    // Root is where to expand, since stop working once fully expanded
    if (root->untried_actions.size() == 0) {
        std::cerr << "[BAD] select_and_expand_workerD only accept non-fully expanded root" << std::endl;
        exit(1);
    }
    if (root->game_over)  // TODO remove this
        return root;
    std::uniform_int_distribution<> dis(0, root->untried_actions.size() - 1);
    const int action_idx = dis(this->rng);
    return root->expand_child_worker(env, action_idx);
}

DecisionNode *MCTS::expand_workerC(Env2048 &env, ChanceNode *root)
{
#ifdef DEBUG2
    std::cerr << "[DEBUG] " << root->id << " In expand workerC" << std::endl;
#endif
    // Starting from root, may need to select down the tree if select result in a old child
    bool expanded = false;
    DecisionNode *cursorD = root->select_child(env, expanded, true);
    while (!cursorD->game_over && !expanded && cursorD->untried_actions.size() == 0) {
        ChanceNode *cursorC = cursorD->select_child(this->explore_c, true);
        if (cursorC == nullptr) {
            return nullptr;
        }
        expanded = false;
        cursorD = cursorC->select_child(env, expanded, true);
    }
    if (cursorD->game_over || expanded)
        return cursorD;
    // Expand at decision node
    std::uniform_int_distribution<> dis(0, cursorD->untried_actions.size() - 1);
    const int action_idx = dis(this->rng);
    return cursorD->expand_child_worker(env, action_idx);
}

// Only called by worker
double MCTS::rollout_worker(Env2048 &env, const DecisionNode *leaf)
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

void MCTS::backpropagate_main(DecisionNode *leaf, double reward) const
{
    MCTSNode *cursor = leaf;
    double min_avg = std::numeric_limits<double>::infinity(),
           max_avg = -std::numeric_limits<double>::infinity();
    while (cursor != nullptr) {
        cursor->stats.update_reward(reward, min_avg, max_avg);
        cursor->fprop.future = false;
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
    MCTSNode *futureRoot = nullptr;
#ifdef DEBUG
    std::cerr << "[MAIN] In run" << std::endl;
#endif
    DecisionNode *leaf = this->select_and_expand_main(futureRoot);
#ifdef DEBUG
    std::cerr << "[MAIN] After select and expand" << std::endl;
#endif
    if (leaf == nullptr) {
        // std::cerr << "[BAD] nullptr" << std::endl;
        return;
    }
    if (!leaf->fprop.future) {
        // Sequential expanded by main
        leaf->fprop.reward = this->rollout_worker(this->main_env, leaf);
        this->backpropagate_worker(futureRoot, leaf, leaf->fprop.reward);
    }
    this->backpropagate_main(leaf, leaf->fprop.reward);
}

void MCTS::run_worker(Env2048 &env, std::shared_ptr<Task> task)
{
    MCTSNode *futureRoot = task->futureRoot;
    // Lock until next isn't null
    std::unique_lock<std::mutex> lock(futureRoot->fprop.mutex);
#ifdef DEBUG2
    std::cerr << "[DEBUG] " << futureRoot->id << " In run worker: " << std::endl;
#endif
    if (task->cancel || futureRoot->fprop.worker_finished.load())
        return;
    // futureRoot working should be true, future should be false
    futureRoot->fprop.worker_processing.store(true);
    futureRoot->fprop.pending_task->cancel = true;
    futureRoot->fprop.pending_task = nullptr;

    bool null_next = false;
    if (futureRoot->fprop.next_step != nullptr)
        lock.unlock();  // Unlock so main can access next_step
    else
        null_next = true;

    while (!futureRoot->fully_expanded_future() && (null_next || futureRoot->fprop.remain_work())) {
    // while (!futureRoot->fully_expanded_future()) {
#ifdef DEBUG2
        std::cerr << "[DEBUG] " << TID << "  " << futureRoot->id << " In while" << std::endl;
#endif
        DecisionNode *leaf = nullptr;
        if (task->is_chance) {
            leaf = this->expand_workerC(env, static_cast<ChanceNode *>(futureRoot));
        } else {
            leaf = this->expand_workerD(env, static_cast<DecisionNode *>(futureRoot));
        }
        if (leaf == nullptr)
            continue;
        leaf->fprop.reward = this->rollout_worker(env, leaf);
        this->backpropagate_worker(futureRoot, leaf, leaf->fprop.reward);
        if (leaf->game_over) {
            continue;
        }

        if (null_next) {
            futureRoot->fprop.next_step = leaf;  // TODO cv?
            futureRoot->fprop.cur_reserve++;
#ifdef DEBUG2
            std::cerr << "[ID] " << futureRoot->id
                << " add next step " << futureRoot->fprop.next_step->id
                << " from null, reserve: " << futureRoot->fprop.cur_reserve << std::endl;
#endif
            lock.unlock();
            null_next = false;
        } else {
            lock.lock();
            DecisionNode *next_ptr = futureRoot->fprop.next_step;
            while (next_ptr != nullptr && next_ptr->fprop.next_step != nullptr) {
#ifdef DEBUG2
                std::cerr << "[DEBUG] " << TID << "  " << futureRoot->id
                    << " Finding next " << next_ptr->id << " " << next_ptr->fprop.next_step->id << std::endl;
#endif
                next_ptr = next_ptr->fprop.next_step;
            }
            futureRoot->fprop.cur_reserve++;
            if (next_ptr == nullptr) {
                futureRoot->fprop.next_step = leaf;  // TODO cv?
#ifdef DEBUG2
                std::cerr << "[ID] " << futureRoot->id
                    << " add next step " << leaf->id
                    << " for root, reserve: " << futureRoot->fprop.cur_reserve << std::endl;
#endif
            }
            else {
                next_ptr->fprop.next_step = leaf;
#ifdef DEBUG2
                std::cerr << "[ID] " << futureRoot->id
                    << " add next step " << leaf->id
                    << " for " << next_ptr->id
                    << ", reserve: " << futureRoot->fprop.cur_reserve << std::endl;
#endif
            }
            lock.unlock();
        }
    }
    if (futureRoot->fully_expanded_future()) {
        futureRoot->fprop.worker_finished.store(true);
    }
    if (null_next) {
        lock.unlock();
    }
    futureRoot->fprop.worker_processing.store(false);
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

ThreadPool::ThreadPool(MCTS &mcts, const unsigned int num_threads)
{
    this->envs.resize(num_threads);
    for (unsigned int i = 0; i < num_threads; i++) {
        this->pool.emplace_back([this, &mcts, i] {
            while (true) {
                bool assigned = false;
                std::shared_ptr<Task> task;
                {
                    std::unique_lock<std::mutex> lock(this->queue_mutex);
#ifdef DEBUG
                    std::cerr << TID << " Waiting" << std::endl;
#endif
                    this->queue_cv.wait(lock, [this] {
                        return (!this->tasks.empty() || this->stop);
                    });
                    if (this->stop)
                        return;
                    if (!this->tasks.empty()) {
#ifdef DEBUG
                        std::cerr << TID << " Get task" << std::endl;
#endif
                        task = this->tasks.front();
                        this->tasks.pop_front();
                        assigned = true;
                    }
                }
                if (assigned)
                    mcts.run_worker(this->envs[i], task);
            }
        });
    }
}

void ThreadPool::stop_all()
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
        this->tasks.emplace_back(task);
    }
    this->queue_cv.notify_one();
}


int mcts_action(const Board &board, const NTupleTD &agent,
                const unsigned int num_threads,
                const double exploration_constant,
                const int iterations, const int rollout_depth)
{
#ifdef DEBUG2
    GLOBAL_ID.store(0);
#endif
    MCTS mcts(board, agent, num_threads, exploration_constant, rollout_depth);
    for (int it = 0; it < iterations; it++) {
#ifdef DEBUG
        std::cerr << "[LOG] Iteration: " << it << std::endl;
#endif
        mcts.run_main();
    }
    mcts.terminate();
    return mcts.get_best_action();
}
