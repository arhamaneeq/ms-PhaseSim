#include "simulator.hpp"

Simulator::Simulator(Grid* grid) : grid(grid), w(grid->getWidth()), h(grid->getHeight()) {
    randStates = genRands(w, h);
    
    J = 0.25;

    MuCrit = - 2.0f * J;
    MuMin = MuCrit - 0.75f;
    MuMax = MuCrit + 0.75f;

    
    TCrit  = 2.0 * J / logf(1 + sqrt(2));
    Tmin = 0;
    Tmax = 2* TCrit;
    
    MuStep = (MuMax - MuCrit) / 25.0f;
    Tstep = (Tmax - Tmin) / 25.0f;
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
void Simulator::incrementTemperature() {
    if (not(T >= Tmax)) T += Tstep;
    if (T > Tmax) T = Tmax;
    std::cout << "T: " << T << std::endl;
}

void Simulator::decrementTemperature() {
    if (not(T <= Tmin)) T -= Tstep;
    if (T < Tmin) T = Tmin;
    std::cout << "T: " <<  T << std::endl;
}

void Simulator::incrementChemPotential() {
    //f /= 255;
    if (not(mu >= MuMax)) mu += MuStep;
    if (mu > MuMax) mu = MuMax;
    std::cout << "u: " <<  mu << std::endl;
}

void Simulator::decrementChemPotential() {
    //f /= 255;
    if (not(mu < MuMin)) mu -= MuStep;
    if (mu < MuMin) mu = MuMin;
    std::cout << "u: " <<  mu << std::endl;
}

void Simulator::step() {
    markovStep(grid->getDeviceData(), w, h, T, mu, randStates, J);
}
