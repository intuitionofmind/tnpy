# Basics of linear algebra

## 1. Vector space

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
**Rank** of a linear map is defined as the dimension of its image $\text{rank}(f)\equiv\text{dim}\left(\text{im}f\right)$.
It is equal to the number of the linearly independent columns as well as rows.
Consider the SVD of the represntative matrix $A=U\Lambda{V}$.
The first $k$ column vectors of $U$ and row vectors of $V$ form a complete basis of the column and row spaces of $A$, respectively.
The rank of $A$ is also equal to the number of non-zero singular values in $\Lambda$.

Furthermore, if $f$ is **one-to-one** as well as **onto** (or surjection), $f$ is called an **isomorphsim** ^[Jim Hefferon, Linear Algebra], which means $V$ is **isomorphic** to $W$ denoted by $V\cong{W}$.
All $n$-dimensional vector spaces are isomorphic to $K^{n}$.
Any isomorphsim is nothing but an element in $\text{GL}(n, K)$.

**Theorem (Rank-nullity theorem)**:
If $f: V\rightarrow{W}$ is a linear map, then
$$
\begin{equation}
    \text{dim}\left(V\right)
    =\text{dim}\left(\text{ker}f\right)+\text{dim}\left(\text{im}f\right).
\end{equation}
$$
Proof:
Since $f$ is linear map, we can directly verify that the elements both in $\text{im}f$ and $\text{ker}f$ satisfy the vector axioms.
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
$\blacksquare$

The vector space spanned by $\left\{\mathbf{h}_{0}, \cdots, \mathbf{h}_{s-1}\right\}$ is called the **orthogonal complement** of $\text{ker}f$.

A map $f: V\to{W}$ is an **antilinear map** ^[1507.06545. Physicists prefer "antilinear" while mathematicans prefer "conjugate linear".] if $f(\alpha{v})=\overline{\alpha}f(v)$ while keeping the linear addition structure.
Antilinear maps usually occur in quantum mechanics.

## 2. Dual and conjugate vector spaces

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

The action of $f$ on $v$ called **contraction** can be interpreted as a homomorphsim: $V^{*}\otimes{V}\to\mathbb{C}$,
$$
\begin{equation}
    f(\mathbf{v})\equiv\langle{f}, \mathbf{v}\rangle
    =\sum_{i}{f}_{i}\mathbf{e}_{i}^{*}\left(\sum_{j}v_{j}\mathbf{e}_{j}\right)
    =\sum_{j}{f}_{j}v_{j}.
\end{equation}
$$

The **conjugate space** of $V$ denoted by $\overline{V}$ is another vector space consisting the same elements and addition structure of $V$.
But the multiplication structure is **antilinear**.
That is, $\forall~v\in\overline{V}/V, \alpha\in\mathbb{C}$, $\alpha*v=\overline{\alpha}\cdot{v}$, where $*$ denotes the multplication on $\overline{V}$ and $\cdot$ denotes the multplication on $V$.
The reason why we introduce this kind of vector space is that we can interprate an antilinear map $f: V\rightarrow{W}$ as a linear map $\overline{V}\rightarrow{W}$ for $f(\alpha*v)=f(\overline{\alpha}\cdot{v})=\alpha{f}(v)$.

Given a vector space isomorphism from its conjugate space to its dual space $g: \overline{V}\rightarrow{V}^{*}$.
$g\in\text{GL}(m, \mathbb{C})$.
$g$ is antilinear on $V\rightarrow\overline{V}^{*}$.
Suppose the basis transforms in a rule as $\mathbf{e}_{i}\rightarrow\sum_{j}{g}_{ij}\mathbf{e}_{j}^{*}$.
For $\mathbf{v}=\sum_{i}v_{i}\mathbf{e}_{i}$, $g\left(\mathbf{v}\right)=\sum_{i}\overline{v}_{i}g\left(\mathbf{e}_{i}\right)=\sum_{ij}\overline{v}_{i}g_{ij}\mathbf{e}_{j}^{*}$.
Once the isomorphism $g$ is defined, the **inner product** of two vectors $\mathbf{u}, \mathbf{v}\in{V}$ can be defined through
$$
\begin{equation}
    g\left(\mathbf{u}, \mathbf{v}\right)
    \equiv{g}\left(\mathbf{u}\right)\left(\mathbf{v}\right)
    =\langle{g}\left(\mathbf{u}\right), \mathbf{v}\rangle
    =\sum_{ij}\overline{u}_{i}g_{ij}\mathbf{e}_{j}^{*}\left(\sum_{k}v_{k}\mathbf{e}_{k}\right)
    =\sum_{ij}\overline{u}_{i}g_{ij}v_{j}.
\end{equation}
$$
in which we require $(g_{ij})$ is a Hermitian positive-definite matrix satisfying $\bar{g}_{ij}=g_{ji}$ and $\text{tr}\left(g_{ij}\right)\neq{0}$.
Note that $\langle{c}\mathbf{u}, \mathbf{v}\rangle=\overline{c}\langle\mathbf{u}, \mathbf{v}\rangle$ and $\langle\mathbf{u}, c\mathbf{v}\rangle=c\langle\mathbf{u}, \mathbf{v}\rangle$.
That is, the complex inner product function is antilinear for the first argument while linear for the second one ^[Some others are **used** to swap th antilinearity and linearity for the two arguments.].
$g(\mathbf{v}, \mathbf{v})=\sum_{i}g_{ii}\vert{v}_{i}\vert^{2}\geqslant{0}$ satisfying the **positive definiteness**.
In addition, we can check
$$
\begin{equation}
   \langle\mathbf{v}, \mathbf{u}\rangle
   =\sum_{ij}\bar{v}_{i}g_{ij}{u}_{j}
   =\sum_{ij}u_{j}\overline{g}_{ji}\overline{v}_{i}
   =\overline{\left(\sum_{ji}\overline{u}_{j}g_{ji}{v}_{i}\right)}
   =\overline{\langle\mathbf{u}, \mathbf{v}\rangle}
\end{equation}
$$
satisfying the so-called **conjugate symmetry**.
Once a vector space is endowed with an inner product, namely an isomporphism from its conjugate space to its dual space is assigned as the metric, it is evaluted to a **normed vector space**.
Distance between vectors there could be defined.
Inner product can be viewed as a kind of homomorphsim: $\overline{V}^{*}\otimes{V}\rightarrow\mathbb{C}$.
Inner product should follow particular axioms (conjugate symmetry, positive definiteness and linearity).
In this sense, we can have another three vector spaces from $V$, namely its dual, conjugate and conjugate dual ones: $V^{*}, \overline{V}, \overline{V}^{*}$.
Physicists often like to denote $\overline{V}^{*}$ as $V^{\dagger}$.

Now let us consider two vector spaces $V$ and $W$.
A linear map $f: V\rightarrow{W}$.
As discussed above, once an isomporphism $g: \overline{V}\rightarrow{V}^{*}$ is given, an inner product on $V$ is defined.
Similarly, we can given another isomporphism $h: \overline{W}\rightarrow{W}^{*}$ to define an inner product on $W$.
Then, the **adjoint** of $f$ denoted by $\tilde{f}$ is defined by $g(\mathbf{v}, \tilde{f}\mathbf{w})=\overline{h(\mathbf{w}, f\mathbf{v})}, \forall~\mathbf{v}\in{V}, \mathbf{w}\in{W}$.
Explicitly,
$\sum_{ij}\overline{v}_{i}g_{ij}\sum_{k}\tilde{f}_{jk}w_{k}=\sum_{ij}w_{i}\overline{h}_{ij}\sum_{k}\overline{f}_{jk}\overline{v}_{k}$.
That is, $\sum_{ijk}\overline{v}_{i}g_{ij}\tilde{f}_{jk}w_{k}=\sum_{ijk}\overline{v}_{i}f_{ij}^{\dagger}h_{jk}^{\dagger}w_{k}$, where $f^{\dagger}\equiv\overline{f}^{T}$.
Thus we find the matrix representation for $\tilde{f}$:
$$
\begin{equation}
    \tilde{f}=g^{-1}f^{\dagger}h^{\dagger}.
\end{equation}
$$

**Lemma**:
If $A$ is a $m\times{n}$ matrix over the field $\mathbb{C}$, $\text{rank}A=\text{rank}A^{\dagger}$.

Proof:
A reduced SVD of $A$ reads $A=U\Lambda{V}$ and $A^{\dagger}=V^{\dagger}\Lambda{U}^{\dagger}$.
$\text{rank}(A)=\text{rank}(A^{\dagger})=\text{dim}(\Lambda)$.
$\blacksquare$

**Lemma**:
Composition of an isomorphism does not change the rank of a map.

Proof:
Suppose $f: V\rightarrow{W}$ is a linear map, $\text{rank}(f)=\text{dim}(\text{im}f)$.
If $g: W\rightarrow{W}^{\prime}$ is an isomorphism, $\forall{\mathbf{w}}=f(\mathbf{v})\in{W}$, $g(\mathbf{w})=\mathbf{w}^{\prime}\in{W}^{\prime}$ is one-to-one correspodent.
That is, $\text{dim}[\text{im}(fg)]=\text{dim}(f)$.
By definition, $\text{rank}(fg)=\text{rank}(f)$.
$\blacksquare$

According to the two Lemmas, since $g$ and $h$ are both isomorphisms, we immediately have $\text{dim}(\text{im}\tilde{f})=\text{rank}\tilde{f}=\text{rank}(g^{-1}f^{\dagger}h^{\dagger})=\text{rank}(f^{\dagger})=\text{rank}f=\text{dim}(\text{im}f)$.

**Theorem (Toy index theorem)**:
$V$ and $W$ are two finite dimensional vector spaces over a field $K$ and $f: V\rightarrow{W}$ is a linear map.
Then
$$
\begin{equation}
    \text{dim}\left(\text{ker}f\right)-\text{dim}\left(\text{ker}\tilde{f}\right)
    =\text{dim}\left(V\right)-\text{dim}\left(W\right).
\end{equation}
$$

Proof:
By the rank-nullity theorem, we have $\text{dim}(V)=\text{dim}(\text{ker}f)+\text{dim}(\text{im}f)$ and $\text{dim}(W)=\text{dim}(\text{ker}\tilde{f})+\text{dim}(\text{im}\tilde{f})$.
Since we already have $\text{dim}(\text{im}f)=\text{dim}(\text{im}\tilde{f})$, immediately we arrive at this conclusion.
$\blacksquare$

**Remarks**:
Note that this theorem says the quantity $\text{dim}\left(V\right)-\text{dim}\left(W\right)$, which is a global property and independent of maps, can be extracted from the details of a specific map $f$.
This can be viewed as a finite-dimensional analogy to the Atiyah–Singer index theorem ^[Mikio Nakahara, Geometry, Topology and Physics, Second Edition.]. 

## 3. Tensors

A dual vector maps a vector to a scalar.
A tensor is a similar mutilinear object mapping some vectors as well as dual vectors to a scalar.
A type-$(p, q)$ tensor $T$ can be defined as a map:
$$
\begin{equation}
    T: \otimes_{i=0}^{p-1}V_{i}^{*}\otimes_{j=0}^{q-1}V_{j}\rightarrow\mathbb{C}.
\end{equation}
$$
Here each vector space represents a **bond** (or **index**) of $T$.
The rank of $T$ is $r=p+q$.
$T$ can be written as a $r$-dimensional **array** over $\mathbb{C}$.
In this sense, we find that a tensor behaviors like a dual vector.
Alternatively, we can also view a tensor as a matrix.
That is, we can group all the indices of $T$ into two groups and the associated vector spaces are divided into **domain** and **codomain**.
Then $T$ behaves like a matrix mapping from the domain to the codomain.
For a simplest example, a linear map $V\rightarrow{W}$ can be represented by a tensor in $V\otimes{W}^{*}$.

**Contraction** of two tensors is a combination of these two linear maps, where the contracted indices are the domain and codomain of the first and second tensors, respectively.
Each pair of indices are nothing but the contraction or action between the dual vector space and its normal one.
Contraction is the basic operation to "glue" tensors together.
Note that the inner product of tensors should be more carefully clarified.

