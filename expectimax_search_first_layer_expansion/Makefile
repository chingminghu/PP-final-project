CXX = g++
CXXFLAGS = -std=c++17 -O2 -I. -I../env -I../TD_learning_sequential_ver

SRCS = play.cpp expectimax_search.cpp ../TD_learning_sequential_ver/n_tuple_TD.cpp ../env/2048env.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET = Expectimax.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

../env/%.o: ../env/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
