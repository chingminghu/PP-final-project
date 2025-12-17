CXX      := mpicxx
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pedantic -Wno-cast-function-type -fopenmp

TARGETS := human_play train_2048 ai_play_2048 train_2048_mpi train_2048_openmp eval_2048

COMMON_SRCS := 2048env.cpp
COMMON_OBJS := $(COMMON_SRCS:.cpp=.o)

AGENT_SRCS  := n_tuple_TD.cpp
AGENT_OBJS  := $(AGENT_SRCS:.cpp=.o)

all: $(TARGETS)

human_play: human_play.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

train_2048: training.o $(COMMON_OBJS) $(AGENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

ai_play_2048: play.o $(COMMON_OBJS) $(AGENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

train_2048_mpi: training_mpi.o $(COMMON_OBJS) $(AGENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

train_2048_openmp: training_openmp.o $(COMMON_OBJS) $(AGENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

eval_2048: eval_2048.o $(COMMON_OBJS) $(AGENT_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

2048env.o: 2048env.cpp 2048env.hpp
	$(CXX) $(CXXFLAGS) -c $<

n_tuple_TD.o: n_tuple_TD.cpp n_tuple_TD.hpp 2048env.hpp
	$(CXX) $(CXXFLAGS) -c $<

human_play.o: human_play.cpp 2048env.hpp
	$(CXX) $(CXXFLAGS) -c $<

training.o: training.cpp 2048env.hpp n_tuple_TD.hpp
	$(CXX) $(CXXFLAGS) -c $<

play.o: play.cpp 2048env.hpp n_tuple_TD.hpp
	$(CXX) $(CXXFLAGS) -c $<

eval_2048.o: eval_2048.cpp 2048env.hpp n_tuple_TD.hpp
	$(CXX) $(CXXFLAGS) -c $<

.PHONY: all clean

clean:
	rm -f *.o $(TARGETS)
