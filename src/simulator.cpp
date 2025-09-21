#include "simulator.hpp"

Simulator::Simulator(Grid* grid) : grid(grid), T(0.5), mu(0.5), w(grid->getWidth()), h(grid->getHeight()) {
    randStates = genRands(w, h);
    
    J = 0.25;

    MuCrit = -2.0f;
    MuMin = -3.0f;
    MuMax = -1.0f;

    TCrit  = 0.567;
    Tmin = 0;
    Tmax = 2* TCrit;
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
void Simulator::incrementTemperature(float f) {
    if (not(T >= Tmax)) T += f;
    if (T > Tmax) T = Tmax;
    std::cout << T << std::endl;
}

void Simulator::decrementTemperature(float f) {
    if (not(T <= Tmin)) T -= f;
    if (T < Tmin) T = Tmin;
    std::cout << T << std::endl;
}

void Simulator::incrementChemPotential(float f) {
    //f /= 255;
    if (not(mu >= MuMax)) mu += f;
    if (mu > MuMax) mu = MuMax;
    std::cout << mu << std::endl;
}

void Simulator::decrementChemPotential(float f) {
    //f /= 255;
    if (not(mu < MuMin)) mu -= f;
    if (mu < MuMin) mu = MuMin;
    std::cout << mu << std::endl;
}

void Simulator::step() {
    markovStep(grid->getDeviceData(), w, h, T, mu, randStates, J);
}