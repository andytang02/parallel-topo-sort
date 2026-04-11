CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

all: graph_gen

graph_gen: graph_gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lstdc++fs

clean:
	rm -f graph_gen

.PHONY: all clean
