#ifndef ENV2048_HPP
#define ENV2048_HPP

#include <vector>

typedef std::vector<std::vector<int>> Board;
typedef std::vector<int> Row;
enum Action {Up, Down, Left, Right};

typedef struct {
    Board board;
    int score;
    bool game_over;
} StepResult;

class Env2048
{
    private:
        int size;
        Board board;
        int score;
        int n_actions;
        bool last_move_valid;

        void add_random_tile();
        Row compress(const Row& row);
        Row merge(const Row& row);
        bool move_left();
        bool move_right();
        bool move_up();
        bool move_down();

    public:
        Env2048(const int size = 4);
        Board reset();
        bool is_game_over() const;
        StepResult step(const int action);
        void print_board() const;

        bool is_move_legal(const int action);
        std::vector<int> get_legal_actions();
        int get_size() const { return size; }
        void set_score(int new_score) { score = new_score; }
        int get_score() const { return score; }
        int get_n_actions() const { return n_actions; }
        bool is_last_move_valid() const { return last_move_valid; }
        void set_board(const Board& new_board) { board = new_board; }
        const Board& get_board() const { return board; }
};

void board_rot90(Board& board);
void board_rot180(Board& board);
void board_rot270(Board& board);

void rgb_set(int r, int g, int b);
void rgb_reset();

#endif