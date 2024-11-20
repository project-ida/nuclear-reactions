---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.15.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/nuclear-reactions/blob/master/fusion-rates.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/nuclear-reactions/blob/master/fusion-rates.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Fusion rates

In this notebook, we'll calculate nuclear fusion rates for using the [Gamow model](https://web.archive.org/web/20200504014928/http://web.ihep.su/dbserv/compas/src/gamow28/eng.pdf). We'll focus our attention the spontaneous fusion of two deuterons in a $\rm D_2$ molecule.

```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
import pandas as pd
pd.set_option('display.max_rows', 300)  # or None to display an unlimited number of rows
```

```python
# Constants
hbar = 1.054571817e-34
amu = 1.66053906892e-27 # 1 amu in kg
a0 = 0.529177210903e-10 # bohr radius in m
Ry_to_eV = 13.605693122990 # 1 Rydberg in eV
J_to_eV = 1/1.602176634e-19 # 1 Joule in eV
```

```python
# Deuterium information
deuteron_mass = 2.013553212745 * amu
muDD = ((deuteron_mass*deuteron_mass) / (deuteron_mass+deuteron_mass)) # reduced mass of D2
```

## Nuclear fusion

Fusion can be described as a two step process:
1. A quantum tunneling event through a potential barrier, with the barrier defined by the interatomic potential between two nuclei.
2. A relaxation of the highly clustered nuclei into some ground state (with the concomitant release of energy)

Step 2 is concerned with nuclear physics and proceeds at a rate $\gamma$ that's determined from experiment and is typically extremely fast (~$10^{20} s^{-1}$).

Step 1 is concerned with solving the Schrödinger equation for the radial wavefunction $R(r)$ that describes the distance, $r$, between the two nuclei:

$$-\frac{\hbar^2}{2\mu} \frac{d^2 R(r)}{dr^2} +  V_{\rm eff}(r) R(r) = E R(r)$$

with $$V_{\rm eff}(r) = V(r) + \frac{L(L+1)\hbar^2}{2\mu r^2}$$

where $\mu$ is the reduced mass of the nuclei, $V(r)$ is the interatomic potential and $\frac{L(L+1)\hbar^2}{2\mu r^2}$ is the centripetal potential associated with the orbital angular momentum, $L$, of the nuclei with respect to one another.

A tunneling probability $T$ can then be calculated based on integrating $|R(r)|^2$ over the classically forbidden region where the two nuclei don't have sufficient energy to overcome the barrier.

The fusion rate can then be simply written as 

$$\Gamma = T\gamma$$




## The Gamow model

Instead of solving the Schrödinger exactly, we can apply the [WKB approximation](https://en.wikipedia.org/wiki/Quantum_tunnelling#WKB_approximation) to obtain a simpler approximate solution. This was the approach taken by [Gamow](https://web.archive.org/web/20200504014928/http://web.ihep.su/dbserv/compas/src/gamow28/eng.pdf) in 1928 which yields an analytical expression for the tunneling probability:

$$T(E) = e^{-2 G} $$

with the Gamow factor $G$ is given by:

$$\int_{r_1}^{r_2} \sqrt{\frac{2\mu}{\hbar^2}\left[V_{\rm eff}(r) - E\right]} \, dr$$

where the integration is inside the classically forbidden region and so $r_1$ and $r_2$ are the classical turning points for the potential barrier.

The Gamow model is designed for 1D problems. However, it can be adjusted to work for realistic 3D problems by augmenting the fusion rate with a "correction factor" C. Specifically:

$$\Gamma = CT\gamma$$

where C depends on the specific fusion problem being considered.

<!-- #region -->
## Potentials

The effective potential includes the interatomic potential and the centripetal potential:

$$V_{\rm eff}(r) = V(r) + \frac{L(L+1)\hbar^2}{2\mu r^2}$$


For a $\rm D_2$ molecule, the interatomic potential $V(r)$ consists of 2 parts - the nuclear and molecular potentials:

$$V(r) = V_{\rm nuc}^{S,L}(r) + V_{\rm mol}(r)$$

### Nuclear potential

We use the Woods-Saxon nuclear potential (in MeV):

$$V_{\rm nuc}^{S,L}(r) ~=~ {V_0 \over 1 + e^{(r - r_S) / a_S}}$$

<!-- #endregion -->

```python
# The nuclear Woods-Saxon nuclear potential
def V_nuc(r, V0, r_S, a_S):
    return V0 / (1 + np.exp((r - r_S) / a_S))
```

The parameters depend on the total spin ($S$) and orbital ($L$) angular momenta of the nuclei as seen below and can be found in [Tomusiak et.al](http://dx.doi.org/10.1103/physrevc.52.1963).

| State | $V_0$ (MeV) | $r_s$ (fm) | $a_s$ (fm) |
|-------|-------------|------------|------------|
| $^1S$ | -74.0       | 1.70       | 0.90       |
| $^5S$ | -15.5       | 3.59       | 0.81       |
| $^3P$ | -13.5       | 5.04       | 0.79       |
| $^5D$ | -15.5       | 3.59       | 0.81       |

```python
# Parameters for nuclear potential depend on the state
states = [
    {"state": r"$^1S$: L=0, S=0", "L":0, "S":0, "V0": -74.0, "r_S": 1.70, "a_S": 0.90},
    {"state": r"$^5S$: L=0, S=2", "L":0, "S":2, "V0": -15.5, "r_S": 3.59, "a_S": 0.81},
    {"state": r"$^3P$: L=1, S=1", "L":1, "S":1, "V0": -13.5, "r_S": 5.04, "a_S": 0.79},
    {"state": r"$^5D$: L=2, S=2", "L":2, "S":2, "V0": -15.5, "r_S": 3.59, "a_S": 0.81},
]
```

We can then see that the attractive nuclear potentials become negligible within  10 fm and  the $^1S$ state looks to be the most favourable for fusion because of its particularly deep potential well.

```python
# Generate r values (distance in fm)
r_fm = np.linspace(0, 10, 500)

# Plot the potential for each state
plt.figure()
for state in states:
    V0, r_S, a_S = state["V0"], state["r_S"], state["a_S"]
    plt.plot(r_fm, V_nuc(r_fm, V0, r_S, a_S), label=state["state"])

# Customise the plot
plt.xlabel("r (fm)", fontsize=12)
plt.ylabel("$V_{\\rm nuc}$ (MeV)")
plt.legend()
plt.tight_layout()
# plt.savefig("nuc-potential-D2.png", dpi=600)
plt.show()
```

### Molecular potential

A $\rm D_2$ molecule exists because of a balance between the electrostatic repulsion between the nuclei and an attraction  between nuclei and the electron cloud. This balance results in a 74 pm equilibrium distance between the deuterons in a gas.

There are several options for representing the molecular potential. For example, the [Morse Potential](https://en.wikipedia.org/wiki/Morse_potential) is popular. For a more accurate potential, however, we will draw upon the work of [Kolos 1986](http://dx.doi.org/10.1063/1.1669836). We parameterised the numerical Kolos potential as:

$$V_{mol}(r) = \frac{2}{r}(1 - b_1r - b_2r^2) e^{-\alpha r^s}$$

with $r$ in units of the Bohr radius ($a_0$), $V_{mol}$ is in Rydbergs and with
- $\alpha = 0.6255121237003474$
- $b_1 = 1.4752572117720800$
- $b_2 = -0.2369829512108492$
- $s = 1.0659864120418940$

We'll make a function that aligns with the nuclear potential in the sense that it takes $r$ in fm and returned potential in MeV.

```python
alpha = 0.6255121237003474
b1 = 1.4752572117720800
b2 = -0.2369829512108492
s = 1.0659864120418940

# Kolos potential in MeV with r in fm
def V_mol(r_fm):
    r = r_fm*1e-15/a0 # fm to a0
    return (2 / r) * (1 - b1 * r - b2 * r**2) * np.exp(-alpha * r**s) * Ry_to_eV / 1e6
```

```python

# Range for r in pm for plotting and fm for the potential functions
r_pm = np.linspace(25, 500, 500)
r_fm = r_pm*1000

# Calculate V_mol in MeV
V_values = V_mol(r_fm)

# Plot
plt.figure()
plt.plot(r_pm, V_values*1e6, label=r"$V_{mol}(r)$")
plt.xlabel(r"$r$ (pm)")
plt.ylabel(r"$V_{mol}$ (eV)")
# plt.savefig("mol-potential-D2.png", dpi=600)
plt.show()
```

### Centripetal potential

The centripetal potential is a pseudo-potential that results from casting the Schrödinger equation into a spherical coordinate system. Its repulsive nature prevents systems of attracting bodies that orbit around each other from collapsing in on themselves. It's given by:

$$V_{\rm cent} = \frac{L(L+1)\hbar^2}{2\mu r^2}$$

where everything is in SI units.

To align with the other potentials, we'll need to create a function that can take $r$ in fm and returns the potential in MeV.

```python
# Centripetal potential in MeV with r in fm
def V_cent(r_fm, L):
    r = r_fm*1e-15
    return (hbar**2 / (2 * muDD)) * (L * (L + 1)) / (r**2) * J_to_eV / 1e6
```

```python

# Generate r values (distance in fm)
r_fm = np.linspace(1, 10, 500)

# Plot the potential for each state
plt.figure()
for state in states:
    L = state["L"]
    plt.plot(r_fm, V_cent(r_fm, L), label=state["state"])


# Customise the plot
plt.xlabel("r (fm)")
plt.ylabel("$V_{\\rm cent}$ (MeV)")
plt.legend()
plt.tight_layout()
# plt.savefig("cent-potential-D2.png", dpi=600)
plt.show()
```

### Total effective potential

Taken all together, we can see both the coulomb barrier and the presence of the attractive nuclear potential on the other side of the barrier.

```python
fig, ax = plt.subplots()

# Generate r values (distance in fm)
r_fm = np.linspace(0.01, 25, 500)

# Add inset with a zoomed-in section
ax_inset = inset_axes(ax, width="35%", height="35%", loc='center right')

# Mark the zoomed-in area on the main plot
mark_inset(ax, ax_inset, loc1=3, loc2=1, fc="none", ec="0.5")

# Calculate the total effective potential and plot it on both the main and inset axes
for state in states:
    V0, r_S, a_S, L = state["V0"], state["r_S"], state["a_S"], state["L"]
    V_eff = V_nuc(r_fm, V0, r_S, a_S) + V_cent(r_fm, L) + V_mol(r_fm)
    ax.plot(r_fm, V_eff, label=state["state"])
    ax_inset.plot(r_fm, V_eff, label=state["state"])


# Customise the main plot
ax.set_xlabel("r (fm)")
ax.set_ylabel("$V_{\\rm nuc} + V_{\\rm mol} + V_{\\rm cent}$ (MeV)")
ax.legend(loc="lower right", fontsize=9.5)
ax.set_ylim(-60,15)
ax.set_xlim(0,25)

# Customise the inset plot
x1, x2, y1, y2 = 5.5, 20, -0.8, 0.8  # specify the limits for the inset
ax_inset.axhline(0, color='gray', linestyle='dashed')
ax_inset.set_xlim(x1, x2)
ax_inset.set_ylim(y1, y2)
ax_inset.set_yticks([0,0.57])
ax_inset.set_xticks([])
ax_inset.set_yticklabels(['0','0.57'])

# plt.savefig("total-potential-D2.png", dpi=600)
plt.show()

```
