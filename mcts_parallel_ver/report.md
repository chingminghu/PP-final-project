## MCTS
### Method
#### MCTS
##### High level idea
MCTS has 4 steps:
1. *Selection* $\rightarrow$ Done by main
2. *Expansion* $\rightarrow$ Done by workers
3. *Rollout* $\rightarrow$ Done by workers
4. *Backpropagation* $\rightarrow$ Done by main

I implemented with *c++* using a pool of `std::thread` and a work queue.
The idea is based on pipelining: *selection* and *backpropagation* are done by the main thread, while *expansion* and *rollout* are done by pool of worker threads.

In the ideal case, the main thread will perform *selection*, then uses the pre-expanded node from worker as *expansion*. Finally perform *backpropagation* using the rollout reward stored in the pre-expanded node by worker.

In the non-ideal case, where there is no pre-expanded node from worker, the main thread will have to perform *expansion* and *rollout* by itself, just as standard sequential MCTS.

##### Implementation detail

### Experiment
#### MCTS
##### Environment
- My PC: Intel I5-13
- CSIE WS7: When the work station workload was about 5%
##### Baseline
The baseline is the sequential version https://github.com/chingminghu/PP-final-project/tree/main/mcts_sequential_ver.

##### Experiment setting
I run the game for 5 times and take the average.
For each game, I measured the average time for MCTS to make decison on the first 100 steps, each step will search for $it$ iterations.
The reason of choosing the first 100 steps is because the search space is the largest at the start of the game, which leads to more stable results.

Below are the parameters:
- `exploration_constant = 1.41`
- `rollout_depth = 5`
- Use the TD trained weight for value approximation.
    - https://drive.google.com/file/d/1PjSe73gAQLxbDxPdH2nTyqTEd_TCgjDy/view?usp=sharing

##### Experiment result
First of all is the score with $it = 4096$, the average base score without search is 8724, tested with 10000 games.
As we can see from chart xxx, MCTS improves the score by over 3 times, and all parallel version has similar score as sequential version, so parallization did not affect its performance on score.

Then the results for 
