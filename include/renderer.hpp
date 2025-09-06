#include <SDL2/SDL.h>

class Renderer {
    public:
        Renderer(unsigned int w, unsigned int h);
        ~Renderer();

        void draw(const unsigned int* buffer);

        unsigned int const getWidth();
        unsigned int const getHeight();
    private:


        SDL_Window* sdlWindow;
        SDL_Renderer* sdlRenderer;
        SDL_Texture* sdlTexture;
};