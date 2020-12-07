<a href="https://colab.research.google.com/github/project-ida/nuclear-reactions/blob/master/02-internuclear-potentials.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/nuclear-reactions/blob/master/02-internuclear-potentials.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Internuclear potentials


This notebook looks at different sections of the internuclear potential between two nuclei, specifically two deuterium nuclei as in the case of a molecule of deuterons D2. 

The three potentials considered are:
1. The numerical Kolos & Wolniewicz potential (and the parametrized Morse potential)
2. The parametrized Coulomb potential
3. The parametrized Woods-Saxon nuclear potential


## Loading libraries and helper functions

```python
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.constants import pi, epsilon_0, e
from matplotlib import pyplot as plt
```

```python
def bohr_to_pm(length):
    return length*5.292e-11*1e12

def pm_to_bohr(length):
    return length/(5.292e-11*1e12)

def hartree_to_ev(energy):
    return energy*27.211
```

## Kolos & Wolniewicz (KW): numerically determined interatomic potential

From Koonin & Nauenberg (KN) 1989: _"we have taken for the diatomic molecular potential V(r) the best available numerical calculation in the Born-Oppenheimer approximation, due to Kolos and Wolniewicz"_

The KW reference is (the numerically determined potential is given in Table III): 
> Kolos, W., & Wolniewicz, L. (1964). Accurate adiabatic treatment of the ground state of the hydrogen molecule. The Journal of Chemical Physics, 41, 3663–3673.

And KN 1989 describe further: _"For 1.1 < r < 3 this potential is well approximated by the Morse potential"_ (Note that 1.1 bohr = 58 pm and 3 bohr = 159 pm)

The Morse potential is a simple parameterized potential. It is usually given as: 

$$V(r)=D_{e}(e^{-2a(r-r_{e})}-2e^{-a(r-r_{e})})$$

```python
kw = pd.read_csv("./data/k-w_potential.csv")
```

```python
def morsepot_au(r): # in atomic units
    a = 0.4
    return (1+0.1745)*(np.exp(-2*a*(r-1.4))-2*np.exp(-a*(r-1.4))) 
    # To match the KW potential with the Morse potential expression in KN 1989 I had to add +1 to the minimum energy 
    # and adjust a accordingly (see https://en.wikipedia.org/wiki/Morse_potential)
```

```python
kw["Rpm"] = bohr_to_pm(kw["R"]) # in metric units
kw["Eev"] = hartree_to_ev(kw["E"]) # in metric units

kw["Morseev"] = morsepot_au(kw["R"])

kw.head()
```

```python
plt.scatter(kw["Rpm"],kw["Eev"],label="KW potential")
plt.plot(kw["Rpm"],hartree_to_ev(kw["Morseev"]),label="Morse potential")
plt.xlabel("internuclear distance [pm]")
plt.ylabel("potential energy [eV]")
plt.title("KW potential from table in paper alongside Morse potential")
plt.legend()
plt.show() 
```

The KW potential is given at irregular intervals. We'd like to interpolate and resample to a sampling rate of 0.1 pm across the 25 to 200 pm range. The result will be stored in dataframe kw_copy.

```python
kw_copy = pd.DataFrame({'Rpm': kw["Rpm"], 'Eev': kw["Eev"]})
kw_copy.index = kw["Rpm"]

kw_new = pd.DataFrame({'Rpm': new_index, 'Eev': 0})
new_index = np.arange(25,200,0.1)
kw_new.index = new_index

kw_copy = kw_copy.reindex(kw_copy.index.union(kw_new.index))\
                 .interpolate(method='index')\
                 .reindex(kw_new.index)

kw_copy.head()
```

```python
plt.scatter(kw["Rpm"],kw["Eev"],label="KW potential from table")
plt.plot(kw_copy["Rpm"],kw_copy["Eev"],label="interpolated KW potential")
plt.xlabel("internuclear distance [pm]")
plt.ylabel("potential energy [eV]")
plt.title("KW potential from table in paper alongside interpolated KW potential")
plt.legend()
plt.show()
```

## Adding a Coulomb potential


As we get to closer proximities, the KW potential is no longer defined and the Coulomb potential which in the D2 case is essentially 1/r dominates.

It is not clear to me at this point how to merge the two potentials, especially near their intersection. KN 1989 write: _"For smaller values of r we fitted the calculated values of V - 1/r to a seven-term Lagrange interpolation formula."_


The electrostatic potential energy $U$ (in J) between two charged particles is given by:

$$U = \frac{q_1 q_2}{4\pi\varepsilon_0 r}$$

where:

$q$ is the charge in C,

$r$ is the distance between the charges in m.

It can be convenient to work with charge in units of the [elementary electric charge](https://en.wikipedia.org/wiki/Elementary_charge) $e = 1.6\times 10^{-19}$ C. In this case the energy can be written as

$$U = \frac{Z_1 Z_2 e^2}{4\pi\varepsilon_0 r}$$

where

$Z$ is the [charge number](https://en.wikipedia.org/wiki/Charge_number).

It can also be convenient to calculate this energy in [electron volts](https://en.m.wikipedia.org/wiki/Electronvolt) (eV). 1eV is the energy acquired by an electron after being accelerated through a voltage of 1V. Hence 

$1\text{eV} = e\text{J} = 1.602176634\times 10^{-19} \text{J}$

and therefore

$$U_{eV} = \frac{Z_1 Z_2 e}{4\pi\varepsilon_0 r}$$

Finally, if we work in [Hartree atomic units (a.u.)](https://en.wikipedia.org/wiki/Hartree_atomic_units), then this expression simplifies to:

$$U_{au} = \frac{Z_1 Z_2}{r}$$


```python
# when working in au then both e and the denominator except r are 1
def coulomb_au(Z_1,Z_2,r):
  U = Z_1*Z_2 / r
  return U
```

```python
Rpm2 = np.arange(1,100,0.1)

Cev =  coulomb_au(1,1,pm_to_bohr(Rpm2)) - 1.9037 # subtract a constant according to K&N
```

```python
hartree_to_ev(1)
```

```python
plt.plot(kw_copy["Rpm"],kw_copy["Eev"]+hartree_to_ev(0.65),label="KW potential")
plt.plot(Rpm2,hartree_to_ev(Cev),label="Coulomb potential only")

plt.ylim(-40,100)
plt.xlim(-0,100)
plt.xlabel("internuclear distance [pm]")
plt.ylabel("potential energy [eV]")
plt.title("Coulomb potential and KW potential")
plt.legend()
plt.show()
```

## Adding a Woods-Saxon nuclear potential


At close range on the order of a few fm the repulsion of the Coulomb potential gets overridden by the attraction of nucleons by the nuclear force. This potential is often modeled through the [Woods-Saxon potential](https://en.wikipedia.org/wiki/Woods%E2%80%93Saxon_potential):

$$V(r)=-{\frac  {V_{0}}{1+\exp({r-R \over a})}}$$

where 

$$R=r_{0}A^{{1/3}}$$

$r_0$, $V_0$ and a are fitted constants; $r$ is the radius from the center of the nucleus; and $A$ is the mass number of the nucleus whose potential is expressed (i.e. the number of nucleons).

The Woods-Saxon (WS) potential is a mean potential that is assumed to reasonably describe the attraction of the nuclear force from a given nucleus with A=number of nucleons.

```python
A = 4 # 4 nucleons in two deuterium atoms
V0 = 28.5e6 #50 # ground state energy see for He-4 https://pdfs.semanticscholar.org/4c24/b2d33f3b3ca2f288897967dc3a6fdb51f1a0.pdf
a = 0.5e-15
r0 = 1.25e-15
R = r0*A**(0.3333)

def woodssaxon_ev(r, V0, R, a):
    return -V0/(1.0+np.exp((r-R)/a))
```

```python
Rpm1 = np.arange(0.0,1,0.0001)

ws = woodssaxon_ev(Rpm1*1e-12, V0, R, a)

plt.figure(figsize=(4,4))
plt.plot(Rpm1, ws/1e6)
plt.xlim(0,0.01)

plt.xlabel("internuclear distance [pm]")
plt.ylabel("potential energy [MeV]")
plt.title("WS potential")
plt.show()
```

## Assembling it all


Finally, all three potentials are assembled in a single dataframe that covers the range of internuclear distances from 0.1 fm to 100 pm. 

```python
rpm_all = np.concatenate((Rpm1,Rpm2))

fullpotential = pd.DataFrame({'Rpm': rpm_all, 'WSev': 0, 'Cev': 0, 'KWev': 0})
```

```python
thisr = fullpotential.loc[fullpotential["Rpm"] < 1,"Rpm"]
thisws = woodssaxon_ev(thisr*1e-12, V0, R, a)

fullpotential.loc[fullpotential["Rpm"] < 1,"WSev"] = thisws
```

```python
thisr = fullpotential.loc[(fullpotential["Rpm"] > 0.0005) & (fullpotential["Rpm"] < 25.0),"Rpm"]
thisc = hartree_to_ev(energyau(1,1,pm_to_bohr(thisr)) - 1.9037)

fullpotential.loc[thisc.index,"Cev"] = thisc
```

```python
plt.plot(fullpotential["Rpm"],fullpotential["WSev"]+fullpotential["Cev"])
plt.xlim(0,0.01)
```

```python
fullpotential["Rpm"] = np.round(fullpotential["Rpm"],4)
kw_copy["Rpm"] = np.round(kw_copy["Rpm"],4)

#1/r+a (3-4 fermis) #uniform charge distribution model. integrated it out to get potential at short distance 

fullpotential = pd.merge(fullpotential, kw_copy, on='Rpm', how='outer')
fullpotential = fullpotential.fillna(0)
fullpotential
```

```python
fullpotential["assembled"] = fullpotential["WSev"]+fullpotential["Cev"]+fullpotential["Eev"]
```

```python
px.line(fullpotential, x="Rpm", y="assembled")
```