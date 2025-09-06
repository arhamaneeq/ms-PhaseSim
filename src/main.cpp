#include <SDL2/SDL.h>
#include <cstdlib>
#include <iostream>

#include "renderer.hpp"

int main(int argc, char const *argv[])
{
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL Init Video Error: "<< SDL_GetError() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    SDL_Window* sdlWindow = SDL_CreateWindow(
        "SimulationWindow",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        800,
        600,
        SDL_WINDOW_SHOWN
    );

    if (!sdlWindow) {
        std::cerr << "SDL CreateWindow Error : " << SDL_GetError() << std::endl;
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }

    Renderer renderer(800, 600, sdlWindow);
    renderer.update();      
    renderer.waitForExit();  

    return 0;
}
