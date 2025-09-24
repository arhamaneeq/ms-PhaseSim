#include "renderer.hpp"
#include <iostream>


Renderer::Renderer(uint16_t w, uint16_t h, SDL_Window* sdlw, uint16_t gridW, uint16_t gridH) 
: w_width(w), w_height(h), sdlWindow(sdlw), sdlRenderer(nullptr) {
    sdlRenderer = SDL_CreateRenderer(sdlWindow, -1, SDL_RENDERER_ACCELERATED);

    if (!sdlRenderer) {
        std::cerr << "SDL CreateRenderer Error : " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(sdlWindow);
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }

    sdlTexture = SDL_CreateTexture(
        sdlRenderer,
        SDL_PIXELFORMAT_RGBA8888,
        SDL_TEXTUREACCESS_STREAMING,
        gridW,
        gridH
    );
}

Renderer::~Renderer() {
    SDL_DestroyRenderer(sdlRenderer);
    SDL_DestroyWindow(sdlWindow);
    if (sdlTexture) SDL_DestroyTexture(sdlTexture);
    SDL_Quit();
}

void Renderer::drawGrid(uint16_t gridW, uint16_t gridH) {
    SDL_SetRenderDrawColor(sdlRenderer, 0, 0, 0, 255);
    SDL_RenderClear(sdlRenderer);
    SDL_Rect dstRect {0, 0, w_width, w_height};
    SDL_RenderCopy(sdlRenderer, sdlTexture, nullptr, &dstRect);
}

void Renderer::drawBox(float x, float y) {
    const int BOX_HEIGHT = 100;
    const int BOX_WIDTH  = 100;
    const int PADDING    = 20;

    const int BOX_TOP_LEFT_from_x = w_width - BOX_WIDTH - PADDING;
    const int BOX_TOP_LEFT_from_y = w_height - BOX_HEIGHT - PADDING;


    SDL_Rect box {BOX_TOP_LEFT_from_x, BOX_TOP_LEFT_from_y, BOX_WIDTH, BOX_HEIGHT};
    SDL_SetRenderDrawColor(sdlRenderer, 20, 20, 70, 255);
    SDL_RenderFillRect(sdlRenderer, &box);

    SDL_SetRenderDrawColor(sdlRenderer, 100, 100, 255, 255);
    SDL_RenderDrawLine(sdlRenderer, BOX_TOP_LEFT_from_x, BOX_TOP_LEFT_from_y + 0.5 * BOX_HEIGHT, BOX_TOP_LEFT_from_x + BOX_WIDTH, BOX_TOP_LEFT_from_y + 0.5 * BOX_HEIGHT);
    SDL_RenderDrawLine(sdlRenderer, BOX_TOP_LEFT_from_x + 0.5 * BOX_WIDTH, BOX_TOP_LEFT_from_y, BOX_TOP_LEFT_from_x + 0.5 * BOX_WIDTH, BOX_TOP_LEFT_from_y + BOX_HEIGHT);
    SDL_SetRenderDrawColor(sdlRenderer, 200, 200, 255, 255);

    const int POINTER_WIDTH = 6;
    const int POINTER_CENTER_x = BOX_TOP_LEFT_from_x + x * BOX_WIDTH - 0.5 * POINTER_WIDTH;
    const int POINTER_CENTER_y = BOX_TOP_LEFT_from_y + (1.0 - y) * BOX_HEIGHT - 0.5 * POINTER_WIDTH;
    SDL_Rect point {POINTER_CENTER_x, POINTER_CENTER_y, POINTER_WIDTH, POINTER_WIDTH};
    SDL_RenderFillRect(sdlRenderer, &point);
}

void Renderer::update(Grid* grid, Simulator* sim) {
    updateTexture(grid->getHostData(), grid->getWidth(), grid->getHeight());
    
    drawGrid(grid->getWidth(), grid->getHeight());
    drawBox(sim->getTemperatureNorm(), sim->getChemPotentialNorm());

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
    SDL_Color color;

    // if (v < threshold) {
    //     double t = static_cast<double>(v) / threshold;
    //     color.r = static_cast<Uint8>(t * 255);
    //     color.g = static_cast<Uint8>(t * 255);
    //     color.b = 255;
    // } else {
    //     double t = static_cast<double>(v - threshold) / (255 - threshold);
    //     color.r = 255;
    //     color.g = static_cast<Uint8>((1 - t) * 255);
    //     color.b = static_cast<Uint8>((1 - t) * 255);
    // }

    color.r = v;
    color.g = v;
    color.b = v;

    // if (v < threshold) {
    //     color.r = 0;
    //     color.g = 0;
    //     color.b = 0;
    // } else {
    //     color.r = 255;
    //     color.g = 255;
    //     color.b = 255;
    // }

    color.a = 255;
    return color;
}

SDL_Color Renderer::heatmapper(const Cell v) const {
    SDL_Color c;

    if (v == false) {
        c.r = 12;
        c.g = 12;
        c.b = 12;
        c.a = 255;
    } else {
        c.r = 200;
        c.g = 200;
        c.b = 255;
        c.a = 255;
    }

    return c;
}

void Renderer::updateTexture(const Cell* data, int gridW, int gridH) {
    void* pixels;
    int pitch;
    SDL_LockTexture(sdlTexture, nullptr, &pixels, &pitch);

    Cell* pixels8 = reinterpret_cast<Cell*>(pixels);
    for (int y = 0; y < gridH; ++y) {
        Uint32* row = reinterpret_cast<Uint32*>(pixels8 + y * pitch);
        for (int x = 0; x < gridW; ++x) {
            SDL_Color c = heatmapper(data[y * gridW + x]);
            row[x] = (c.r << 24) | (c.g << 16) | (c.b << 8) | c.a;
        }
    }

    SDL_UnlockTexture(sdlTexture);
}