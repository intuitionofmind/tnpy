
# Introduction

## Basic concepts

### Category
A **category** $\mathbf{C}$ consists of following data:
- A set $\text{Ob}(\mathbf{C})=\{A, B, C...\}$, of which elements are called **objects** of $\mathbf{C}$.
- For every pair of objects $A, B\in\text{Ob}(\mathbf{C})$, there is a set $\text{Hom}_{\mathbf{C}}(A, B)$, of which elements are called **morphisms** from $A$ to $B$.
  At this time, $A$ is called **domain** and $B$ is called **codomain**.
- **Composition** of morphisms: for any $A, B, C\in\text{Ob}(\mathbf{C})$, $f: A\rightarrow{B}$ and $g: B\rightarrow{C}$, then $g\circ{f}: A\rightarrow{C}$.
  Given another $h: C\rightarrow{D}$, the associativity means $h\circ(g\circ{f})=(h\circ{g})\circ{f}$.
- For each $A\in\text{Ob}(\mathbf{C})$, there is an **identity morphism** $\text{id}_{A}: A\rightarrow{A}$.

They are subjected to:
$$
\begin{equation}
    f\circ\text{id}_{A}=\text{id}_{B}\circ{f}=f, \quad
    g\circ{f}: A\rightarrow{C},
\end{equation}
$$
for any $f: A\rightarrow{B}$, $g: B\rightarrow{C}$ and $h: C\rightarrow{D}$.
They are always refered as **identity laws** and **associativity**, respectively.

### Functor
Let $\mathbf{C}$ and $\mathbf{D}$ be two categories.
A **functor** $F: \mathbf{C}\rightarrow\mathbf{D}$ is a map.


## Graphical language

In a graphical language ^[0908.3347, Peter Selinger, A survey of graphical languages for monoidal categories.], objects are represented by **wires** and morphisms are represented by **nodes**.
This is the tensor language.
Composition of morphisms are represented by the tensor contraction.
This was first used by Penrose ^[1912.10049, Jacob Biamonte, Lectures on Quantum Tensor Networks.] and is refered as **Pennrose graphical notation**.

In the category $\mathbf{Vect}_{\mathbb{K}}$, the category of finite dimensional vector spaces, objects are vector spaces.
The wires (bonds) must be **directional**.
We use incoming and outgoing arrows to distinguish normal vector spaces and dual ones, respectively.
In the case $\mathbb{K}=\mathbb{R}$, since there is no difference between normal and dual vector spaces, arrows are always dropped.


# Monoidal catetory

For a (finite) quantum many-body system on a lattice, we always assume a **tensor product** structure saying that the total Hilbert space is the tensor product of all local Hilbert spaces.
Naively speaking, "tensor product" here just means placing all pieces parallelly.
Quantum entanglement can be produced among them.
Mathematically, this is described by the so-called **Cartesian product**.