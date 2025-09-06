#include "renderer.hpp"
#include <iostream>

Renderer::Renderer(uint16_t w, uint16_t h, SDL_Window* sdlw) 
: w_width(w), w_height(h), sdlWindow(sdlw), sdlRenderer(nullptr) {
    sdlRenderer = SDL_CreateRenderer(sdlWindow, -1, SDL_RENDERER_ACCELERATED);

    if (!sdlRenderer) {
        std::cerr << "SDL CreateRenderer Error : " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(sdlWindow);
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }
}

Renderer::~Renderer() {
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(sdlWindow);
    SDL_Quit();
}

void Renderer::drawGrid() {
    SDL_SetRenderDrawColor(sdlRenderer, 50, 50, 50, 255);
    SDL_RenderClear(sdlRenderer);
}

void Renderer::drawBox() {
    SDL_Rect box {w_width - 120, w_height - 60, 100, 50};
    SDL_SetRenderDrawColor(sdlRenderer, 255, 255, 255, 255);
    SDL_RenderFillRect(sdlRenderer, &box);
}

void Renderer::update() {
    drawGrid();
    drawBox();

    SDL_RenderPresent(sdlRenderer);
}

void Renderer::waitForExit() {
    SDL_Event e;            // 1
    bool quit = false;      // 2
    while (!quit) {         // 3
        while (SDL_PollEvent(&e)) {  // 4
            if (e.type == SDL_QUIT)  // 5
                quit = true;         // 6
        }
        SDL_Delay(16);      // 7
    }
}


SDL_Color Renderer::heatmapper(uint8_t v, uint8_t threshold) const {
    if (v > 255) {v = 255;}
    if (v < 0)   {v = 0;}

    SDL_Color color;
    
    // Coolwarm
    if (v < threshold) {
        double t = v / threshold;
        color.r = static_cast<Uint8>(t * 255);
        color.g = static_cast<Uint8>(t * 255);
        color.b = 255;
    } else {
        double t = (v - 0.5) / 0.5;
        color.r = 255;
        color.g = static_cast<Uint8>((1 - t) * 255);
        color.b = static_cast<Uint8>((1 - t) * 255);
    }

    color.a = 255;

    return color;
}