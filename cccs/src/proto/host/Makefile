

SOURCES = $(wildcard *.proto)
TARGET = $(patsubst %.proto,%_pb2.py,$(SOURCES))

all: $(TARGET)

UNAME_S = $(shell uname -p)
ifeq ($(UNAME_S), x86_64)
	CMD = python3 -m grpc_tools.protoc
endif
ifeq ($(UNAME_S), aarch64)
	CMD = python37 -m grpc_tools.protoc
endif
ifeq ($(UNAME_S), armv7l)
	CMD = protoc
endif

	
%_pb2.py: %.proto
	$(CMD) --python_out=.  --grpc_python_out=.  -I. $^

clean:
	rm -rf $(TARGET)