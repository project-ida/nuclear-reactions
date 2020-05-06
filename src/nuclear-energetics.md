---
jupyter:
  jupytext:
    formats: ipynb,src//md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<a href="https://colab.research.google.com/github/project-ida/nuclear/blob/matt-sandbox/nuclear-energetics.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; <a href="https://nbviewer.jupyter.org/github/project-ida/nuclear/blob/matt-sandbox/nuclear-energetics.ipynb" target="_parent"><img src="https://nbviewer.jupyter.org/static/img/nav_logo.svg" alt="Open In nbviewer" width="100"/></a>


# Energetics of nuclear reactions


This notebook is about the energy involved in nuclear processes. We take a look at
1. Binding energy
2. Alpha decay

```python
import pandas as pd
pd.set_option('precision', 10)

import matplotlib.pyplot as plt
%matplotlib inline
```

## Binding energy


Binding energy is the amount of energy you need to put into a nucleus in order to separate it into its constituent protons and neutrons.

There are many places where you can find data about the binding energy for different nuclei - we will look at two of them.


### PNPI - [Petersburg Nuclear Physics Institute](http://dbserv.pnpi.spb.ru/)


A PDF of the binding energy data can be found [here](http://dbserv.pnpi.spb.ru/elbib/tablisot/toi98/www/astro/table2.pdf). We have reformatted the data into a csv file to make it easier to analyse.

```python
# source data http://dbserv.pnpi.spb.ru/elbib/tablisot/toi98/www/astro/table2.pdf
pnpi = pd.read_csv("./data/binding-energies-pnpi.csv",header=7)
```

```python
pnpi.head()
```

The PNPI data above contains the following columns of data:
- **A** - Atomic mass number
- **EL** - Element label
- **BE (MeV)** - Binding energy in MeV


It is instructive to calculate the binding energy per nucleon - this gives us a sense of which nuclei are particularly stable.

```python
pnpi["BE/A (MeV)"] = pnpi["BE (MeV)"]/pnpi["A"]
```

```python
pnpi.plot.scatter(x="A", y="BE/A (MeV)",figsize=(15,8), title="Binding energy per nucleon    (Fig 1)");
```

The most stable element is the one with maximum binding energy per nucleon - it is Ni-62

```python
most_stable_index = pnpi["BE/A (MeV)"].argmax()
pnpi.loc[most_stable_index]
```

The hump shape of Fig 1 indicates that energy can be released from nuclear fusion of light elements and nuclear fission of heavy elements. 

The current data only goes up to atomic mass 135. To go further we need to look at a different dataset


### IAEA nuclear data services - [Atomic Mass Data Center](https://www-nds.iaea.org/amdc/)


A txt file containing the binding energy data can be found [here](https://www-nds.iaea.org/amdc/ame2016/mass16.txt). We have reformatted the data into a csv file to make it easier to analyse and excluded non-experimental values (denoted by # in the original txt file).

```python
# source data https://www-nds.iaea.org/amdc/ame2016/mass16.txt
iaea = pd.read_csv("./data/binding-energies-iaea.csv",header=13)
```

```python
iaea.head()
```

The IAEA data above contains the following columns of data:
- **N** - Number of neutrons
- **Z** - Number of protons
- **A** - Atomic mass number
- **EL** - Element label
- **DEL_M (keV)** - [Mass excess](https://en.wikipedia.org/wiki/Mass_excess) in keV (technically this should be $keV/c^2$ but the $c^2$ factor is often dropped)
- **BE/A (keV)** - Binding energy per nucleon in keV
- **Mass (mu-u)** - Atomic mass in millionths of a standard atomic mass unit ([Dalton](https://en.wikipedia.org/wiki/Dalton_(unit)))


Let's first renormalise our units from `keV` into `MeV` for energy and from `mu-u` to `u` for mass units. (*n.b. we don't do this in the csv file because of the [precision limitations](https://docs.python.org/3/tutorial/floatingpoint.html) of the resulting floating point numbers*)

```python
iaea["BE/A (keV)"] = iaea["BE/A (keV)"]/1000
iaea["DEL_M (keV)"] = iaea["DEL_M (keV)"]/1000
iaea["Mass (mu-u)"] = iaea["Mass (mu-u)"]/1000000
iaea.rename(columns={'BE/A (keV)': 'BE/A (MeV)', 'DEL_M (keV)': 'DEL_M (MeV)', 'Mass (mu-u)': 'Mass (u)'}, inplace=True)
```

```python
iaea.head()
```

```python
iaea.plot.scatter(x="A", y="BE/A (MeV)",figsize=(15,8), title="Binding energy per nucleon    (Fig 2)");
```

Fig 2 is much more similar to what we see in text books at school. 

We can also check to see whether the IAEA data agrees with PNPI about the most stable element.

```python
most_stable_index = iaea["BE/A (MeV)"].argmax()
iaea.loc[most_stable_index]
```

$^{62}Ni$ again - lovely.

We are now going to use the IAEA data to look at the energetics of alpha decay.


## Alpha decay


### What is an alpha particle?


[Alpha decay](https://en.wikipedia.org/wiki/Alpha_decay) is a process that involves an unstable parent nucleus "spitting out" a He-4 nucleus (aka an alpha particle). Let's have a look at the alpha particle and see if we can understand the iaea data for it.

To extract only the alpha particle entry (with N=2 and Z=2) from the iaea table, we can use the [`query`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.query.html) function from pandas

```python
alpha = iaea.query("N==2 & Z==2")
alpha
```

Although `DEL_M (MeV)`, `BE/A (MeV)` and `Mass (u)` might appear like independent quantities, they are in fact intimately related to each other, let's see how.


Mass is the most familiar to us so we'll start there. We normally think of helium as having a mass of 4, associated with the number of nucleons from which it is made (2 protons and 2 neutrons), but this simple picture is not the whole story. 

In [atomic mass units](https://en.wikipedia.org/wiki/Dalton_(unit%29) `u` (aka Dalton), The mass of helium in is 4.002603254.

Compare this with the mass of 2 protons and 2 neutrons:

```python
proton = iaea.query("N==0 & Z==1")
neutron = iaea.query("N==1 & Z==0")
```

```python
2*proton["Mass (u)"].values[0] + 2*neutron["Mass (u)"].values[0]
```

There is a difference between the two values (a [mass defect](https://en.wikipedia.org/wiki/Nuclear_binding_energy#Mass_defect)), the mass of helium is less than the mass of its constituent protons by an amount:

```python
mass_defect = 4.0329798960000005 - 4.002603254
mass_defect
```

or, energy units $1u = 931.494MeV$,

```python
u = 931.494
```

```python
mass_defect*u
```

This 28.2957MeV is the energy needed to split the helium apart into its protons and neutrons, i.e. this is the binding energy. Per nucleon this is exactly what appears in the `BE/A (MeV)` column of the iaea table.

```python
alpha_binding_energy = mass_defect*u/4
alpha_binding_energy
```

It is sometimes convenient to think about the mass of a nucleus in terms of how much it deviate from the simple picture given by the number of nucleons. For Heluium this would be:

```python
4.002603254 - 4
```

or, energy units,

```python
(4.002603254 - 4)*u
```

This 2.4249MeV is called Mass excess and is what appears in the in the `DEL_M (MeV)` column of the iaea table.


### Spontaneous alpha decay 


Alpha decay creates a daughter nucleus with 2 fewer protons and 2 fewer neutrons than the parent. 

Let's look at a real life example. The alpha decay of [uranium-238](https://en.wikipedia.org/wiki/Uranium-238).

```python
U238 = iaea.query("N==146 & Z==92")
U238
```

Subtracting the alpha nucleus from the U-238 gives us an element with the following number of neutrons and protons:

```python
daughter_NZ = U238[["N","Z"]] - alpha[["N","Z"]].values[0]
daughter_NZ
```

What is this element? We need to query the iaea table.

```python
Th234 = iaea.query("N==144 & Z==90")
Th234
```

The alpha decay product of uranium-238 is apparently thorium-234. This decay can (and indeed does) happen spontaneously in nature because of the positive mass difference between the U-238 and the products (Th-234  +alpha). Let's see this explicitly:

```python
mass_defect = U238["Mass (u)"].values[0] - (Th234["Mass (u)"].values[0] + alpha["Mass (u)"].values[0])
mass_defect
```

U-238 has more mass than the Th-234 + alpha. This mass difference is converted to kinetic energy of the products (with most going to the alpha because it's much lighter than Th).

We can therefore expect that the alpha particle will be released with the following kinetic energy (in MeV):

```python
mass_defect*u
```

This is indeed what is reported (see "Decay modes" in [U-238 wiki entry](https://en.wikipedia.org/wiki/Uranium-238))


### Induced alpha decay


In addition to decay processes that happen spontaneously, we can also imagine exciting nuclei into higher energy states from which they are then energetically able to decay. [Photodisintegration](https://en.wikipedia.org/wiki/Photodisintegration), [Photofission](https://en.wikipedia.org/wiki/Photofission) and [Neutron activation](https://en.wikipedia.org/wiki/Neutron_activation) are examples of such a situation.

Another important example that we'll now look at is the [breeding of tritium from lithium](https://en.wikipedia.org/wiki/Tritium#Lithium).


Most of the lithium in the world is Li-7

```python
Li7 = iaea.query("N==4 & Z==3")
Li7
```

If we are to imagine the possibility of Li-7 undergoing alpha decay then, it's daughter nucleus would have the following numbers of neutrons and protons:

```python
daughter_NZ = Li7[["N","Z"]] - alpha[["N","Z"]].values[0]
daughter_NZ
```

This daughter element is H-3, otherwise knows as tritium

```python
H3 = iaea.query("N==2 & Z==1")
H3
```

However, if we look at the difference in energy (in MeV) between Li-7 and products (i.e H-3 + alpha):

```python
(Li7["Mass (u)"].values[0]  - (H3["Mass (u)"].values[0] + alpha["Mass (u)"].values[0]))*u
```

We see that it's negative i.e. spontaneous decay is not energetically possible.

We can however create an excited form of Li-7 by "bombarding" Li-6 with neutrons.

```python
Li6 = iaea.query("N==3 & Z==3")
Li6
```

The energy difference between reactants (Li-6 + n) and products (H-3 + alpha) is then:

```python
( Li6["Mass (u)"].values[0] + neutron["Mass (u)"].values[0] - 
 (H3["Mass (u)"].values[0] + alpha["Mass (u)"].values[0]) )*u
```

So, 4.78MeV of energy is released when we combine Li-6 and a neutron - there is therefore no need to "bombard" the Li-6 with very high energy neutrons, apparently any energy will do.

We can play these kind of energy comparison games for many different scenarios in order to hunt for possible novel reactions. It is helpful to be able to do these comparisons across many elements at once - this is what we will finish with.


### Automation


We are now going to extend some of the code used when looking at the alpha decay of an individual nucleus. This will allow us to analyse all the elements in one go. In a sense we will be gathering together many hypothetical reactions from which we can later select/discard according to energy conservation criteria.


We start as we did before by subtracting the alpha nucleus, but this time from **all** elements in the iaea list. This gives us daughter nuclei with the following number of neutrons and protons:

```python
daughter_NZ = iaea[["N","Z"]] - alpha[["N","Z"]].values[0]
```

```python
daughter_NZ.head(10)
```

Some of the rows in the above table don't make sense because they are either both zero or contain negative numbers. The first row that does make sense is row number 9.

Although we can tell by eye that the daughter element corresponding the row 9 with N=1 and Z=0 is the neutron, in general we will need to query the iaea table to find this out. This querying is similar to what we did earlier, the only difference is that below we now use [fstrings](https://realpython.com/python-f-strings/) to demonstrate how we go about removing the hard coded numbers.

```python
q = f"N=={daughter_NZ.loc[9]['N']} & Z=={daughter_NZ.loc[9]['Z']}"
q
```

```python
iaea.query(q)
```

We are now ready to automate the process of finding the daughter elements (for those that have one).

```python
parent_index = []
daughter_index = []
for i, row in daughter_NZ.iterrows():
    try:
        q = f"N=={row['N']} & Z=={row['Z']}"
        daughter_index.append(iaea.query(q).index[0])
        parent_index.append(i)
    except:
        continue
```

```python
parents_alpha_decay = iaea.loc[parent_index]
parents_alpha_decay.reset_index(inplace=True, drop=True)

daughters_alpha_decay = iaea.loc[daughter_index]
daughters_alpha_decay.reset_index(inplace=True, drop=True)
```

```python
parents_alpha_decay.head()
```

```python
daughters_alpha_decay.head()
```

The above two tables pair the parents and daughters together. We can now calculate the kinetic energy of the hypothetical decay reactions.

```python
parents_alpha_decay["E_kin (MeV)"] = (parents_alpha_decay["Mass (u)"] - 
                      (daughters_alpha_decay["Mass (u)"] + alpha["Mass (u)"].values[0]))*u
```

```python
parents_alpha_decay.head(10)
```

```python
parents_alpha_decay.plot.scatter(x="A", y="E_kin (MeV)",figsize=(15,8), 
                                 title="Kinetic energy of alpha decay products     (Fig 3)");
plt.plot((270, 0), (0, 0), 'r-');
```

Fig 3 shows us that on the whole (with the exception of He-5, Li-5 and Be-8) spontaneous alpha decay is only energetically possible when the mass number gets higher than about 100. We can see this explicitly by querying the `parents_alpha_decay` table

```python
parents_alpha_decay.query("`E_kin (MeV)` > 0")
```

We can also see that if we are able to deposit more energy (in MeV) than

```python
abs(parents_alpha_decay["E_kin (MeV)"].min())
```

Then, energetically speaking, alpha decay is possible for all elements.


Some elements, require a lot less than 25MeV. For example, we can use the query function to pick out  [Palladium](https://en.wikipedia.org/wiki/Palladium) and [Silver](https://en.wikipedia.org/wiki/Silver)

```python
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15,8), sharey=True)
parents_alpha_decay.query("EL=='Pd'").plot.scatter(x="A", y="E_kin (MeV)", ax=axes[0],
                                    title="Kinetic energy of Pd alpha decay products     (Fig 4)");
parents_alpha_decay.query("EL=='Ag'").plot.scatter(x="A", y="E_kin (MeV)",ax=axes[1],
                                    title="Kinetic energy of Ag alpha decay products     (Fig 5)");
```

For the most abundant type of palladium (Pd-108):

```python
parents_alpha_decay.query("EL=='Pd' & A==108")
```

We would need to provide at least 3.9MeV of energy to make alpha decay energetically possible.

In contrast, Silver (whose most abundant isotope is Ag-107):

```python
parents_alpha_decay.query("EL=='Ag' & A==107")
```

requires only 2.8MeV.


## Next up...

Energetics is just one of the factors to consider when hunting for possible novel nuclear reactions. We will explore some of the other factors in the next notebook.

```python

```
