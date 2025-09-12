#pragma once

#include <cstdint>
#include "grid.hpp"
#include "wrapper.cuh"
#include <iostream>

class Simulator {
    using Cell = uint8_t;

    public:
        Simulator(Grid* grid);
        ~Simulator();

        void setTemperature(double temp);
        void setChemPotential(double _mu);

        double getTemperature() const;
        double getChemPotential() const;

        void incrementTemperature(float f);
        void incrementChemPotential(float f);
        void decrementTemperature(float f);
        void decrementChemPotential(float f);

        void step();

    private:
        Grid* grid;
        float T, mu;
        int w, h;
        curandState* randStates;

        float TCrit, Tmin, Tmax;
        float MuCrit, MuMin, MuMax;
};