#include "grid.hpp"
#include <cstdint>


Grid::Grid(uint16_t w, uint16_t h) 
: width(w), height(h), d_cells(nullptr), h_cells(nullptr) {
    size_t size = static_cast<size_t>(width) * height;

    h_cells = new Cell[size];
    std::fill(h_cells, h_cells + size, 127);
    d_cells = (Cell*) allocateDeviceMemory(size * sizeof(Cell));
};

Grid::~Grid() {
    delete[] h_cells;
    deallocateDeviceMemory(d_cells);
}

void Grid::syncToDevice() {
    size_t size = static_cast<size_t>(width) * height;
    copyMemory(d_cells, h_cells, size * sizeof(Cell), cudaMemcpyHostToDevice);
}

void Grid::syncToHost() {
    size_t size = static_cast<size_t>(width) * height;
    copyMemory(h_cells, d_cells, size * sizeof(Cell), cudaMemcpyDeviceToHost);
}

uint8_t* Grid::getDeviceData() {
    return d_cells;
}

uint8_t* Grid::getHostData() const {
    return h_cells;
}

uint16_t Grid::getWidth() const {
    return width;
}

uint16_t Grid::getHeight() const {
    return height;
}