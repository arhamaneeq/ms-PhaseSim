# Compilers
CXX := g++
NVCC := nvcc

# Compiler flags
CXXFLAGS := -Iinclude -std=c++17 -O2 -Wall
NVCCFLAGS := -Iinclude -O2 -std=c++17
LDFLAGS := -lSDL2

# Directories
SRC_DIR := src
BUILD_DIR := build

# Sources
CPP_SRCS := $(wildcard $(SRC_DIR)/*.cpp)
CU_SRCS  := $(wildcard $(SRC_DIR)/*.cu)

# Objects
CPP_OBJS := $(patsubst $(SRC_DIR)/%.cpp,$(BUILD_DIR)/%.o,$(CPP_SRCS))
CU_OBJS  := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%.o,$(CU_SRCS))
OBJS := $(CPP_OBJS) $(CU_OBJS)

# Target executable
TARGET := ./phase

# Default target
all: $(TARGET)

# Link all objects
$(TARGET): $(OBJS)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(OBJS) -o $@ $(LDFLAGS)

# Compile C++ sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile CUDA sources
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cu
	@mkdir -p $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Clean build
clean:
	rm -rf $(BUILD_DIR)/*
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean
