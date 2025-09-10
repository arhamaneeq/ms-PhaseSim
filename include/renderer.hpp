#pragma once

#include <SDL2/SDL.h>
#include <cstdint>

#include "grid.hpp"
#include "simulator.hpp"

class Renderer {
    public:
        Renderer(uint16_t w, uint16_t h, SDL_Window* sdlWindow);
        ~Renderer();

        void update(Grid* grid, Simulator* sim);

        void drawGrid(uint16_t gridW, uint16_t gridH);
        void drawBox();
        void waitForExit();

        uint16_t getWidth() const;
        uint16_t getHeight() const;

        void updateTexture(const uint8_t* data, int gridW, int gridH);

    private:
        uint16_t w_width, w_height; // dimensions for window
        uint16_t b_width, b_height; // dimensions for tiny box thingy

        SDL_Window* sdlWindow;
        SDL_Renderer* sdlRenderer;
        SDL_Texture* sdlTexture;

        SDL_Color heatmapper(const uint8_t v, uint8_t threshold) const;
};