#pragma once

#include <cstdint>

class Grid {
    public:
        Grid(uint16_t w, uint16_t h);
        ~Grid();

        unsigned int getWidth();
        unsigned int getHeight();

        const uint8_t* hostData();

    private:
        uint16_t width, height;

        uint8_t* d_cells; // device / for cuda
        uint8_t* h_cells; // host
};