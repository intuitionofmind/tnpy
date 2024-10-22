# Basics of linear algebra

##  Vector space

Refs:

* Jim Hefferon, Linear Algebra, Fourth Edition
* Mikio Nakahara, Geometry, Topology and Physics, Second Edition

A **vector space** (or **linear space**) $V$ over a field $K$ $(\mathbb{R}$ or $\mathbb{C}$) is a set, in which two operations: addition and mutiplication by an element of $K$, are defined.
For $\mathbf{u}, \mathbf{v}, \mathbf{w}\in{V}$ and $c, d\in{K}$, they should obey the so-called eight vector axioms ^[Mikio Nakahara, Geometry, Topology and Physics, Second Edition] below:

1. $\mathbf{u}+\mathbf{v}=\mathbf{v}+\mathbf{u}$.
2. $(\mathbf{u}+\mathbf{v})+\mathbf{w}=\mathbf{u}+(\mathbf{v}+\mathbf{w})$.
3. There exists a zero vector $\mathbf{0}$ such that $\mathbf{v}+\mathbf{0}=\mathbf{v}$.
4. $\forall\mathbf{u}\in{V}$, there exists $-\mathbf{u}$ such that $\mathbf{u}+(-\mathbf{u})=\mathbf{0}$.
5. $c(\mathbf{u}+\mathbf{v})=c\mathbf{u}+c\mathbf{v}$.
6. $(c+d)\mathbf{u}=c\mathbf{u}+d\mathbf{u}$.
7. $(cd)\mathbf{u}=c(d\mathbf{u})$.
8. $1\mathbf{u}=\mathbf{u}$.

Here $1$ is the unit element of $K$.

Let $\{\mathbf{v}_{i}\}$ be a set of $k$ vectors from $V$.
$i=0, \dots, k-1$.
If the equation $\sum_{i=0}^{k-1}x_{i}\mathbf{v}_{i}=0$ only have one trivial solution, namely $x_{i}=0, i=0\dots{k-1}$, $\{\mathbf{v}_{i}\}$ is called **linearly independent**.
If any $\mathbf{v}\in{V}$ can be uniquiely written as a linear combination of a set of linearly independent vectors $\{\mathbf{e}_{i}\}$, namely $v=\sum_{i=0}^{n-1}v_{i}\mathbf{e}_{i}$, $\{\mathbf{e}_{i}\}$ is called a basis of $V$.
$n$ is the dimension of $V$.

Given two vector spaces $V$ and $W$, a map $f: V\to{W}$ is a **linear map** if
$$
\begin{equation}
    f(a_{0}\mathbf{v}_{0}+a_{1}\mathbf{v}_{1})
    =a_{0}f(\mathbf{v}_{0})+a_{1}f(\mathbf{v}_{1}), ~\forall{a}_{0}, a_{1}\in{K}; ~\forall\mathbf{v}_{0}, \mathbf{v}_{1}\in{V}.
\end{equation}
$$
Linear map is a **homomorphsim** ^[from Greek words *homoios morphe*, means “similar form”.] preserving baisic structures (addition and scalar multiplication) of the vector space.
If $W=K$, $f$ is called a **linear function**.
The **image** of $f$ is $\text{im}f\equiv{f}(V)\subset{W}$.
The **kernel** of $f$ is $\text{ker}f\equiv\{\mathbf{v}\in{V}\vert{f}(\mathbf{v})=\mathbf{0}\}$.

Furthermore, if $f$ is **one-to-one** as well as **onto** (or surjection), $f$ is called an **isomorphsim** ^[Jim Hefferon, Linear Algebra], which means $V$ is **isomorphic** to $W$ denoted by $V\cong{W}$.
All $n$-dimensional vector spaces are isomorphic to $K^{n}$.
Any isomorphsim is nothing but an element in $\text{GL}(n, K)$.

**Theorem**: If $f: V\rightarrow{W}$ is a linear map, then
$$
\begin{equation}
    \text{dim}\left(V\right)
    =\text{dim}\left(\text{ker}f\right)+\text{dim}\left(\text{im}f\right).
\end{equation}
$$
Proof: Since $f$ is linear map, we can directly verify that the elements both in $\text{im}f$ and $\text{ker}f$ satisfy the vector axioms.
Therefore,  $\text{ker}f$ and $\text{im}f$ are vector spaces, with dimension $r$ and $s$, respectively.
Suppose $\left\{\mathbf{g}_{0}, \cdots, \mathbf{g}_{r-1}\right\}$ and $\left\{\mathbf{h}_{0}^{\prime}, \cdots, \mathbf{h}_{s-1}^{\prime}\right\}$ are the basis for $\text{ker}f$ and $\text{im}f$.
$f(\mathbf{h}_{i})=\mathbf{h}_{i}^{\prime}, i=0, \cdots, s-1$.
If $\mathbf{v}\in{V}$, then $f(\mathbf{v})=\sum_{i}\alpha_{i}\mathbf{h}_{i}^{\prime}=\sum_{i}\alpha_{i}f(\mathbf{h}_{i})$.
Since $f$ is a linear map, $f(\mathbf{v}-\sum_{i}\alpha_{i}\mathbf{h}_{i})=\mathbf{0}$.
That is, $\mathbf{v}-\sum_{i}\alpha_{i}\mathbf{h}_{i}=\sum_{j}\beta_{j}\mathbf{g}_{j}\in\text{ker}f$.
$\mathbf{v}=\sum_{i}\alpha_{i}\mathbf{h}_{i}+\sum_{j}\beta_{j}$ is spanned by $\left\{\mathbf{g}_{0}, \cdots, \mathbf{g}_{r-1}, \mathbf{h}_{0}, \cdots, \mathbf{h}_{s-1}\right\}$.
Next, let us check the linear independence of these vectors.
Assume $\sum_{i}a_{i}\mathbf{g}_{i}+\sum_{j}b_{j}\mathbf{h}_{j}=\mathbf{0}$, we have $f(\mathbf{0})=\mathbf{0}=\sum_{i}a_{i}f(\mathbf{g}_{i})+\sum_{j}b_{j}f(\mathbf{h}_{j})=\sum_{j}b_{j}f(\mathbf{h}_{j})=\sum_{j}b_{j}\mathbf{h}_{j}^{\prime}$ since $f(\mathbf{g}_{i})=\mathbf{0}$.
Note that $\left\{\mathbf{h}_{0}^{\prime}, \cdots, \mathbf{h}_{s-1}^{\prime}\right\}$ is a linearly independent set, which implies $b_{j}=0, \forall{j}$.
Then, $\sum_{i}a_{i}\mathbf{g}_{i}=\mathbf{0}$ and $a_{i}=0, \forall{i}$.
Therefore, $\left\{\mathbf{g}_{0}, \cdots, \mathbf{g}_{r-1}, \mathbf{h}_{0}, \cdots, \mathbf{h}_{s-1}\right\}$ is indeed a linearly independent basis for $V$.
$\text{dim}(V)=r+s$.

The vector space spanned by $\left\{\mathbf{h}_{0}, \cdots, \mathbf{h}_{s-1}\right\}$ is called the **orthogonal complement** of $\text{ker}f$.

A map $f: V\to{W}$ is an **antilinear map** ^[1507.06545. Physicists prefer "antilinear" while mathematicans prefer "conjugate linear".] if $f(\alpha{v})=\overline{\alpha}f(v)$ while keeping the linear addition structure.
Antilinear maps usually occur in quantum mechanics.

## Dual and conjugate vector spaces

Now let us set $K=\mathbb{C}$.
$K=\mathbb{R}$ is only a special case.
Suppose $V(m, \mathbb{C})$ is a vector space over the field $\mathbb{C}$.
The set of all linear functions $f: V\to\mathbb{C}$ also comprised of another vector space.
This vector space denoted by $V^{*}(m, \mathbb{C})$ is called the **dual vector space** of $V(m, \mathbb{C})$.
Suppose $V$ is spanned by $\left\{\mathbf{e}_{0}, \cdots, \mathbf{e}_{n-1}\right\}$.
If $\mathbf{v}=\sum_{i}c_{i}\mathbf{e}_{i}$, $f(\mathbf{v})=\sum_{i}c_{i}f(\mathbf{e}_{i})$.
$\mathbf{e}_{i}^{*}\equiv{f}(\mathbf{e}_{i})$ is a basis of $V^{*}$.
Furthermore, it can be chosen arbitrarily as $\mathbf{e}_{i}^{*}(\mathbf{e}_{j})\equiv\delta_{ij}$.
Any $f=\sum_{i}f_{i}\mathbf{e}_{i}^{*}\in{V}^{*}$.

The action of $f$ on $v$ can be interpreted as **contraction** as a homomorphsim: $V^{*}\otimes{V}\to\mathbb{C}$,
$$
\begin{equation}
    f(\mathbf{v})\equiv\langle{f}, \mathbf{v}\rangle
    =\sum_{i}{f}_{i}\mathbf{e}_{i}^{*}\left(\sum_{j}v_{j}\mathbf{e}_{j}\right)
    =\sum_{j}{f}_{j}v_{j}.
\end{equation}
$$

The **complex conjugate space** of $V$ denoted by $\overline{V}$ is another vector space consisting the same elements and addition structure of $V$.
But the multiplication structure is antilinear.
That is, $\forall~v\in\overline{V}/V, \alpha\in\mathbb{C}$, $\alpha*v=\overline{\alpha}\cdot{v}$, where $*$ denotes the multplication on $\overline{V}$ and $\cdot$ denotes the multplication on $V$.
The reason why we introduce this kind of vector space is that we can interprate an antilinear map $f: V\rightarrow{W}$ as a linear map $\overline{V}\rightarrow{W}$ for $f(\alpha*v)=f(\overline{\alpha}\cdot{v})=\alpha{f}(v)$.

Given a vector space isomorphism $g: \overline{V}\rightarrow{V}^{*}$.
$g\in\text{GL}(m, \mathbb{C})$.
Basis transforms in the rule $\mathbf{e}_{i}\rightarrow\sum_{j}{g}_{ij}\mathbf{e}_{j}^{*}$.
Note that $g$ is antilinear on $V\rightarrow{V}^{*}$ as discussed above.
That is, given $\mathbf{v}=\sum_{i}v_{i}\mathbf{e}_{i}$, $g\left(\mathbf{v}\right)=\sum_{i}\overline{v}_{i}g\left(\mathbf{e}_{i}\right)=\sum_{ij}\overline{v}_{i}g_{ij}\mathbf{e}_{j}^{*}$.
Once this isomorphism is defined, the **inner product** of two vectors $\mathbf{v}, \mathbf{w}\in{V}$ can be defined through
$$
\begin{equation}
    g\left(\mathbf{v}\right)\left(\mathbf{w}\right)
    =\langle{g}\left(\mathbf{v}\right), \mathbf{w}\rangle
    =\sum_{ij}\overline{v}_{i}g_{ij}\mathbf{e}_{j}^{*}\left(\sum_{k}w_{k}\mathbf{e}_{k}\right)
    =\sum_{ij}\overline{v}_{i}g_{ij}w_{j}.
\end{equation}
$$
in which we require $(g_{ij})$ is a Hermitian positive-definite matrix satisfying $\bar{g}_{ij}=g_{ji}$ and $\text{tr}\left(g_{ij}\right)\neq{0}$.
Note that $\langle{c}\mathbf{v}, \mathbf{w}\rangle=\overline{c}\langle\mathbf{v}, \mathbf{w}\rangle$ and $\langle\mathbf{v}, c\mathbf{w}\rangle=c\langle\mathbf{v}, \mathbf{w}\rangle$.
That is, the complex inner product function is antilinear for the first argument while linear for the second one.
$g(\mathbf{v}, \mathbf{v})=\sum_{i}g_{ii}\vert{v}_{i}\vert^{2}\geqslant{0}$ satisfying the **positive definiteness**.
In addition, we check
$$
\begin{equation}
   \langle\mathbf{w}, \mathbf{v}\rangle
   =\sum_{ij}\bar{w}_{i}g_{ij}{v}_{j}
   =\sum_{ij}v_{j}\overline{g}_{ji}\overline{w}_{i}
   =\overline{\left(\sum_{ji}\overline{v}_{j}g_{ji}{w}_{i}\right)}
   =\overline{\langle\mathbf{v}, \mathbf{w}\rangle}.
\end{equation}
$$
satisfying the so-called **conjugate symmetry**.
Once a vector space is endowed with an inner product, namely an isomporphism from its conjugate space to its dual space is assigned as the metric, it is evaluted to a **normed vector space**.
Distance between vectors there could be defined.
Inner prodcut can be viewed as a homomorphsim or contraction $\overline{V}^{*}\otimes{V}\rightarrow\mathbb{C}$.

In this sense, we can have another three vector spaces from $V$, namely its dual, conjugate and dual contra
## Tensors

