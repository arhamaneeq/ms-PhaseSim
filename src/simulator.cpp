#include "simulator.hpp"

Simulator::Simulator(Grid* grid) : grid(grid), T(0.5), mu(0.5) {
    //
}

void Simulator::setTemperature(double temp) {
    T = temp;
}

void Simulator::setChemPotential(double _mu) {
    mu = _mu;
}

double Simulator::getTemperature() const {
    return T;
}

double Simulator::getChemPotential() const {
    return mu;
}