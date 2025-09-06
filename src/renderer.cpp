#include "renderer.hpp"

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