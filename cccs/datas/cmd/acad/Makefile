# 
# Reference arithmetic coding
# Copyright (c) Project Nayuki
# 
# https://www.nayuki.io/page/reference-arithmetic-coding
# https://github.com/nayuki/Reference-arithmetic-coding
# 


CXXFLAGS += -std=c++11 -O1 -Wall -Wextra -fsanitize=undefined


.SUFFIXES:

.SECONDARY:

.DEFAULT_GOAL = all
.PHONY: all clean


#OBJ = ArithmeticCoder.o BitIoStream.o FrequencyTable.o 
#MAINS = ArithmeticCompress

OBJ = ArithmeticCoder.o BitIoStream.o FrequencyTable.o 
MAINS = ArithmeticCompress

all: $(MAINS)

clean:
	rm -f -- $(OBJ) $(MAINS:=.o) $(MAINS)
	rm -rf .deps

%: %.o $(OBJ)
	$(CXX) $(CXXFLAGS) -o $@ $^

%.o: %.cpp .deps/timestamp
	$(CXX) $(CXXFLAGS) -c -o $@ -MMD -MF .deps/$*.d $<

.deps/timestamp:
	mkdir -p .deps
	touch .deps/timestamp

-include .deps/*.d
