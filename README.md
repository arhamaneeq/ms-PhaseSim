# Metrpolis MCMC Simulation of Vapour-Phase Model

## Model

We use a grand canonical ensemble, i.e., the control parameters used are temperature $T$, chemical potential $\mu$ and volume $V$. Note that due to the finite lattice in our simulation, $V$ is inherently fixed, with the fluctuating parameters being energy $E$ and $N$.

$$
P(E, N) \propto e^{-\beta(E - \mu N)} \text{  where  } \beta = \frac{1}{k_B T}
$$

This weight function will be enforced by the Metropolis Markov-Chain Monte Carlo method.

### Acceptance Rule

A proposed move will be accepted with probability $p_\text{accept} = \text{min} (1, e^{-\beta \Delta \Phi})$ where $\Delta\Phi$ is the change in thermodynamic potential, which in the grand canonical ensemble is $\Delta(E - \mu  N)$, therefore the acceptance rule is

$$
p_\text{accept} = \text{min}(1, e^{-\beta[\Delta E - \mu \Delta N]})
$$

### Moves

A site is picked at random, a move $n_i \rightarrow n_i \pm 1$ (respecting bounds: 0 - 255) is proposed. We compute $\Delta N = \pm 1$ and compute $\Delta E$ from our Hamiltonian, and accept the move with probability $p_\text{accept}$.

