#include <SDL2/SDL.h>
#include <cstdlib>
#include <iostream>

#include "renderer.hpp"
#include "renderer.hpp"
#include "simulator.hpp"
#include "grid.hpp"

int main(int argc, char const *argv[])
{
    const int simWidth  = 500;
    const int simHeight = 500;

    const int winWidth  = 1200;
    const int winHeight = 800;

    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        std::cerr << "SDL Init Video Error: "<< SDL_GetError() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    SDL_Window* sdlWindow = SDL_CreateWindow(
        "SimulationWindow",
        SDL_WINDOWPOS_CENTERED,
        SDL_WINDOWPOS_CENTERED,
        winWidth,
        winHeight,
        SDL_WINDOW_SHOWN
    );

    if (!sdlWindow) {
        std::cerr << "SDL CreateWindow Error : " << SDL_GetError() << std::endl;
        SDL_Quit();
        std::exit(EXIT_FAILURE);
    }

    Renderer renderer(winWidth, winHeight, sdlWindow, simWidth, simHeight);
    Grid grid(simWidth, simHeight);
    Simulator simulator(&grid);
    simulator.setTemperature(0.5);
    simulator.setChemPotential(-2.0f);

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                quit = true;
            } else if (e.type == SDL_KEYDOWN) {
                std::cout << SDL_GetKeyName(e.key.keysym.sym) << std::endl;
                switch (e.key.keysym.sym) {
                    case SDLK_ESCAPE:
                        quit = true;
                        break;
                    case SDLK_UP:
                        simulator.incrementChemPotential();
                    break;
                    case SDLK_DOWN:
                        simulator.decrementChemPotential();
                    break;
                    case SDLK_RIGHT:
                        simulator.incrementTemperature();
                    break;
                    case SDLK_LEFT:
                        simulator.decrementTemperature();
                        break;
                    default:
                        std::cout << SDL_GetKeyName(e.key.keysym.sym) << std::endl;
                }
            }
        }

        grid.syncToDevice();
        simulator.step();
        grid.syncToHost();
        renderer.update(&grid, &simulator);

        SDL_Delay(16);
    }

    

    return 0;
}
