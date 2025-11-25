CXX = g++
CXXFLAGS = -std=c++17 -O2 -I. -I../env

SRCS = training.cpp n_tuple_TD.cpp ../env/2048env.cpp
OBJS = $(SRCS:.cpp=.o)

TARGET = TD_learning.exe

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) -o $@

../env/%.o: ../env/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) $(TARGET)
