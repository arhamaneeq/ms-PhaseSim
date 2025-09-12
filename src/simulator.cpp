#include "simulator.hpp"

Simulator::Simulator(Grid* grid) : grid(grid), T(0.5), mu(0.5), w(grid->getWidth()), h(grid->getHeight()) {
    randStates = genRands(w, h);
    T = 0.02;
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
    if (not(T >= Tmax)) T += f;
    std::cout << T << std::endl;
}

void Simulator::decrementTemperature(double f) {
    if (not(T <= Tmin)) T -= f;
    std::cout << T << std::endl;
}

void Simulator::incrementChemPotential(double f) {
    if (not(mu >= MuMax)) mu += f;
    std::cout << mu << std::endl;
}

void Simulator::decrementChemPotential(double f) {
    if (not(mu >= MuMax)) mu += f;
    std::cout << mu << std::endl;
    mu -= f;
}

void Simulator::step() {
    markovStep(grid->getDeviceData(), w, h, T, mu, randStates);
}