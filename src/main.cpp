#include <SDL2/SDL.h>
#include <cstdlib>
#include <iostream>

#include "renderer.hpp"

int main(int argc, char const *argv[])
{
    Renderer renderer(800, 600);
    renderer.update();      
    renderer.waitForExit();  

    return 0;
}
