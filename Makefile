CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

NVCC     = nvcc
NVFLAGS  = -std=c++17 -O2 -arch=sm_75 -ccbin=/usr/bin/g++-11

all: topo_sort

topo_sort: topo_sort.cu
	$(NVCC) $(NVFLAGS) -o $@ $<

# standalone graph generator (writes to disk, optional)
graph_gen: graph_gen.cpp
	$(CXX) $(CXXFLAGS) -o $@ $< -lstdc++fs

clean:
	rm -f graph_gen topo_sort

.PHONY: all clean
