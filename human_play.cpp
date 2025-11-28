#include "2048env.hpp"
#include <iostream>
#include <ctime>

int main(void){
    srand(static_cast<unsigned int>(time(nullptr)));
    Env2048 env;
    env.reset();
    std::cout << "Welcome to 2048!\n";
    std::cout << "You can use the following commands to play:\n";
    std::cout << "  - 'w' to move up\n";
    std::cout << "  - 's' to move down\n";
    std::cout << "  - 'a' to move left\n";
    std::cout << "  - 'd' to move right\n";
    std::cout << "  - 'q' to quit the game\n\n";
    std::cout << "Initial Board:\n";
    env.print_board();
    while(true) {
        char command;
        std::cout << "Enter command: ";
        std::cin >> command;

        int action;
        switch(command) {
            case 'w': 
                action = Up;   
                break;
            case 's':  
                action = Down;
                break;
            case 'a': 
                action = Left;
                break;
            case 'd': 
                action = Right; 
                break;
            case 'q': 
                std::cout << "Thanks for playing!\n";
                return 0;
            default:
                std::cout << "Invalid command. Please try again.\n";
                continue;
        }

        StepResult result = env.step(action);
        env.print_board();
        
        if(result.game_over) {
            std::cout << "Game Over! Your final score is: " << result.score << "\n";
            break;
        }
    }
}