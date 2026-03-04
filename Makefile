CXXFLAGS := -Wall -Wextra -Werror -pedantic -std=c++20 -fopenmp
RELEASEFLAGS := -Ofast

MAIN_SRCS := simulation.cc common.cc
BONUS_SRCS := simulation.cc common.cc
HEADERS := common.h

OBJS := $(MAIN_SRCS:.cc=.o)

.PHONY: all bonus clean

all: sim.debug sim.perf
bonus: bonus.debug bonus.perf

# Debug build
sim.debug: $(MAIN_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DDEBUG -g $(MAIN_SRCS) -o $@

# Perf (release) build
sim.perf: $(MAIN_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(RELEASEFLAGS) $(MAIN_SRCS) -o $@

# Bonus Debug build
bonus.debug: $(BONUS_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) -DDEBUG -g $(BONUS_SRCS) -o $@

# Bonus Perf (release) build
bonus.perf: $(BONUS_SRCS) $(HEADERS)
	$(CXX) $(CXXFLAGS) $(RELEASEFLAGS) $(BONUS_SRCS) -o $@

clean:
	rm -f sim.debug sim.perf bonus.debug bonus.perf
