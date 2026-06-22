HAPLY_SDK := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))sdk
CXX       ?= g++

all: inverse3_server

inverse3_server: inverse3_bridge.cpp
	$(CXX) -std=c++17 -O2 -pthread \
	    -I$(HAPLY_SDK)/include \
	    inverse3_bridge.cpp \
	    $(HAPLY_SDK)/libHaply.HardwareAPI.a \
	    -o inverse3_server -pthread

clean:
	rm -f inverse3_server

.PHONY: all clean
