# Metrpolis MCMC Simulation of Vapour-Phase Model

## How to Use

Once the simulation Window opens, use the arrow keys to change the simulation paramters.

- `RIGHT_ARROW` / `LEFT_ARROW` : Increase / Decrease Temperature, $T$
- `UP_ARROW` / `DOWN_ARROW` : Increase / Decrease Chemical Potential $\mu$
- `C` : Return to criticality $(\mu_{\text{Crit}}, T)$
- `P` : Write current state to terminal

### What You Should be Able to Observe

#### Phase Change at Low $T$ and varying $\mu$

If T is low (but non-zero) you should see the system stabilise as either fully vapourous or liquid. If you increase/decrease $\mu$ past $\mu_\text{Crit}$, then you should expect to see gradual condensation/boiling.

#### Metastability at Minimum $T$

If $T = 0$, then moving $\mu$ past $\mu_\text{Crit}$ will not induce phase change. Instead the system enters a metastable state, where any deviation from $T = 0$ will rapidly induce phase change.

#### Criticality

At $(\mu_\text{Crit}, T)$ the system is at a critical state, where neither the gaseous or liquid phases dominate. 

## Model

We use a grand canonical ensemble, i.e., the control parameters used are temperature $T$, chemical potential $\mu$ and volume $V$. Note that due to the finite lattice in our simulation, $V$ is inherently fixed, with the **fluctuating** parameters being energy $E$ and $N$.

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

Note now, that *a site is picked at random* carries probability of its own, and thus we need to be careful how this is implemented in massively parallel code. The canonical implementation of MCMC requires one cell being worked on at a time. If we proposed one move for every cell at the same time in parallel, then the validity of the MCMC breaks since we use $\Delta E$ s that depend on neighbouring states which are themselves changing. 

The work around here is to parallelly propose moves on an alternating checkerboard lattice, such that no adjacent neighbours have the same colour.

