#include "simulator.hpp"

Simulator::Simulator(Grid* grid) : grid(grid), T(0.5), mu(0.5), w(grid->getWidth()), h(grid->getHeight()) {
    randStates = genRands(w, h);
}

Simulator::~Simulator() {
    deallocateDeviceMemory(randStates);
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
void Simulator::incrementTemperature(double f) {
    T += f;
}

void Simulator::decrementTemperature(double f) {
    T -= f;
}

void Simulator::incrementChemPotential(double f) {
    mu += f;
}

void Simulator::decrementChemPotential(double f) {
    mu -= f;
}

void Simulator::step() {
    markovStep(grid->getDeviceData(), w, h, T, mu, randStates);
}