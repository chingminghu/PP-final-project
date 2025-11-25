#include "2048env.hpp"
#include <iostream>
#include <algorithm>
#include <utility>
#include <ctime>

Env2048::Env2048(const int size)
{
    srand(static_cast<unsigned int>(time(nullptr)));
    
    this->size = size;
    this->n_actions = 4;
    this->last_move_valid = true;

    reset();
}

Board Env2048::reset()
{
    this->board = Board(size, Row(size, 0));
    this->score = 0;
    add_random_tile();
    add_random_tile();
    return board;
}

void Env2048::add_random_tile()
{
    std::vector<std::pair<int, int>> empty_tiles;
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if(board[i][j] == 0) {
                empty_tiles.emplace_back(std::make_pair(i, j));
            }
        }
    }
    if(empty_tiles.empty())
        return;
    int random_index = rand() % empty_tiles.size();
    int value = (rand() % 10 == 0) ? 4 : 2;
    board[empty_tiles[random_index].first][empty_tiles[random_index].second] = value;
    return;
}

Row Env2048::compress(const Row& row)
{
    Row new_row;
    for(int i = 0; i < size; i++) {
        if(row[i] != 0) {
            new_row.push_back(row[i]);
        }
    }
    while(new_row.size() < size) {
        new_row.push_back(0);
    }
    return new_row;
}

Row Env2048::merge(const Row& row)
{
    Row new_row = row;
    for(int i = 0; i < size - 1; i++) {
        if(new_row[i] != 0 && new_row[i] == new_row[i + 1]) {
            new_row[i] *= 2;
            score += new_row[i];
            new_row[i + 1] = 0;
        }
    }
    return new_row;
}

bool Env2048::move_left()
{
    bool moved = false;
    for(int i = 0; i < size; i++) {
        Row original_row = board[i];
        Row new_row = compress(original_row);
        new_row = merge(new_row);
        new_row = compress(new_row);
        if(new_row != original_row) {
            moved = true;
            board[i] = new_row;
        }
    }
    return moved;
}

bool Env2048::move_right()
{
    board_rot180(board);
    bool moved = move_left();
    board_rot180(board);
    return moved;
}

bool Env2048::move_up()
{
    board_rot90(board);
    bool moved = move_left();
    board_rot270(board);
    return moved;
}

bool Env2048::move_down()
{
    board_rot270(board);
    bool moved = move_left();
    board_rot90(board);
    return moved;
}

bool Env2048::is_game_over() const
{
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            if(board[i][j] == 0) {
                return false;
            }
            if(i < size - 1 && board[i][j] == board[i + 1][j]) {
                return false;
            }
            if(j < size - 1 && board[i][j] == board[i][j + 1]) {
                return false;
            }
        }
    }
    return true;
}

StepResult Env2048::step(const int action)
{
    bool moved = false;
    switch(action) {
        case Up:
            moved = move_up();
            break;
        case Down:
            moved = move_down();
            break;
        case Left:
            moved = move_left();
            break;
        case Right:
            moved = move_right();
            break;
        default:
            throw std::invalid_argument("Invalid action");
    }

    last_move_valid = moved;
    Board before_board = board;
    if(moved)
        add_random_tile();
    
    bool done = is_game_over();
    return {before_board, score, done};
}

bool Env2048::is_move_legal(const int action)
{
    Board original_board = board;
    int original_score = score;
    bool moved = false;
    switch(action) {
        case Up:
            moved = move_up();
            break;
        case Down:
            moved = move_down();
            break;
        case Left:
            moved = move_left();
            break;
        case Right:
            moved = move_right();
            break;
        default:
            throw std::invalid_argument("Invalid action");
    }
    if(moved) {
        board = original_board;
        score = original_score;
    }
    return moved;
}

std::vector<int> Env2048::get_legal_actions()
{
    std::vector<int> legal_actions;
    for(int action = 0; action < n_actions; action++) {
        if(is_move_legal(action)) {
            legal_actions.push_back(action);
        }
    }
    return legal_actions;
}

void Env2048::print_board() const
{
    for(const auto& row : board) {
        for(const auto& tile : row) {
            switch(tile) {
                case 0:
                    rgb_set(128, 128, 128); // Grey for empty tiles
                    break;
                case 2:
                    rgb_set(238, 228, 218); // Light beige for 2
                    break;
                case 4:
                    rgb_set(237, 224, 200); // Beige for 4
                    break;
                case 8:
                    rgb_set(242, 177, 121); // Orange for 8
                    break;
                case 16:
                    rgb_set(245, 149, 99); // Dark orange for 16
                    break;
                case 32:
                    rgb_set(246, 124, 95); // Red for 32
                    break;
                case 64:
                    rgb_set(246, 94, 59); // Dark red for 64
                    break;
                case 128:
                    rgb_set(237, 207, 114); // Yellow for 128
                    break;
                case 256:
                    rgb_set(237, 204, 97); // Light yellow for 256
                    break;
                case 512:
                    rgb_set(237, 200, 80); // Dark yellow for 512
                    break;
                case 1024:
                    rgb_set(237, 197, 63); // Gold for 1024
                    break;
                case 2048:
                    rgb_set(237, 194, 46); // Bright gold for 2048
                    break;
                default:
                    rgb_set(60,60,60); // Dark grey for larger numbers
            }
            std::cout << tile << "\t";
        }
        std::cout << "\n";
    }
    rgb_set(255, 255, 255);
    std::cout << "Score: " << score << "\n\n";
    rgb_reset();
    return;
}

void board_rot90(Board& board)
{
    int size = board.size();
    Board temp(size, Row(size, 0));
    for(int i = 0; i < size; i++) {
        for(int j = 0; j < size; j++) {
            temp[size - 1 - j][i] = board[i][j];
        }
    }
    board = temp;
    return;
}

void board_rot180(Board& board)
{
    board_rot90(board);
    board_rot90(board);
    return;
}

void board_rot270(Board& board)
{
    board_rot90(board);
    board_rot90(board);
    board_rot90(board);
    return;
}

void rgb_set(int r, int g, int b)
{
    std::cout << "\033[1;38;2;" << r << ";" << g << ";" << b << "m";
    return;
}

void rgb_reset()
{
    std::cout << "\033[0m";
    return;
}