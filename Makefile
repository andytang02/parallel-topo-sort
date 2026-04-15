CXX      = g++
CXXFLAGS = -std=c++17 -O2 -Wall -Wextra

NVCC     = nvcc
NVFLAGS  = -std=c++17 -O2 -arch=sm_75 -ccbin=/usr/bin/g++-11

# Shared library object files (compiled by g++, linked into both binaries)
LIB_OBJS = graph_gen_lib.o seq_topo_sort.o

all: topo_sort topo_sort_omp

graph_gen_lib.o: graph_gen_lib.cpp graph.h graph_gen.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

seq_topo_sort.o: seq_topo_sort.cpp graph.h graph_gen.h seq_topo_sort.h
	$(CXX) $(CXXFLAGS) -c -o $@ $<

topo_sort: topo_sort.cu $(LIB_OBJS) graph.h graph_gen.h seq_topo_sort.h
	$(NVCC) $(NVFLAGS) -o $@ topo_sort.cu $(LIB_OBJS)

topo_sort_omp: topo_sort_omp.cpp $(LIB_OBJS) graph.h graph_gen.h seq_topo_sort.h
	$(CXX) $(CXXFLAGS) -fopenmp -o $@ topo_sort_omp.cpp $(LIB_OBJS)

# Standalone graph generator (writes to disk, optional)
graph_gen: graph_gen.cpp graph_gen_lib.o graph.h graph_gen.h
	$(CXX) $(CXXFLAGS) -o $@ graph_gen.cpp graph_gen_lib.o -lstdc++fs

clean:
	rm -f graph_gen topo_sort topo_sort_omp $(LIB_OBJS)

.PHONY: all clean