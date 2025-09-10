#pragma once

#include <cstdint>
#include "grid.hpp"
#include "wrapper.cuh"

class Simulator {
    using Cell = uint8_t;

    public:
        Simulator(Grid* grid);
        ~Simulator();

        void setTemperature(double temp);
        void setChemPotential(double _mu);

        double getTemperature() const;
        double getChemPotential() const;

        void incrementTemperature(double f);
        void incrementChemPotential(double f);
        void decrementTemperature(double f);
        void decrementChemPotential(double f);

        void step();

    private:
        Grid* grid;
        double T, mu;
        int w, h;
        curandState* randStates;
};