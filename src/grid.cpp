#include "grid.hpp"
#include <cstdint>


Grid::Grid(uint16_t w, uint16_t h) 
: width(w), height(h), d_cells(nullptr), h_cells(nullptr) {
    size_t size = static_cast<size_t>(width) * height;

    h_cells = new Cell[size];
    std::fill(h_cells, h_cells + size, 0);
    d_cells = (Cell*) allocateDeviceMemory(size);
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