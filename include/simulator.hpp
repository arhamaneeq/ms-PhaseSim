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

        float getTemperatureNorm() const;
        float getChemPotentialNorm() const;

        void incrementTemperature();
        void incrementChemPotential();
        void decrementTemperature();
        void decrementChemPotential();

        void toCritical();

        void step();

    private:
        Grid* grid;
        float T, mu;
        int w, h;

        curandState* randStates;

        float J;
        float TCrit, Tmin, Tmax, Tstep;
        float MuCrit, MuMin, MuMax, MuStep;
};