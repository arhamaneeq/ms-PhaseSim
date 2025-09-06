#pragma once

#include <cstdint>
#include <malloc.h>
#include <stdexcept>
#include "wrapper.cuh"

class Grid {
    using Cell = uint8_t;

    public:
        Grid(uint16_t w, uint16_t h);
        ~Grid();

        uint16_t getWidth() const;
        uint16_t getHeight() const;

        const Cell* hostData() const;

        void syncToDevice();
        void syncToHost();

    private:
        uint16_t width, height;

        Cell* d_cells; // device / for cuda
        Cell* h_cells; // host

        uint16_t& index(uint16_t i, uint16_t j);
};