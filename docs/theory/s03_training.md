# Training theory

## Learning the solution with fixed time-step

Consider the generic non-linear autonomous ODE system :

$\frac{dU}{dt} = f(U)$

where $U$ is a vector that can represent a $n$-dimensional problem state
(_i.e_ solution).
A **solution-predicting** NN-model is built to take an input solution 
$U_0 = U(t_0)$ and give as output the solution $U_1$ such that

$U_1 \simeq U(t_0 + \Delta t)$

## Learning the update with fixed time-step

Using the same autonomous ODE system, a **update-predicting** NN-model
with scaling factor $\alpha$
is built to take an input solution $U_0 = U(t_0)$
and give as output the update $\Delta_U$ such that

$\Delta_U \simeq \alpha\left(U(t_0 + \Delta t) - U_0\right)$

A natural suggestion for the scaling is $\alpha=\Delta t^{-1}$,
such that the next solution can be computed like this :

$U_1 = U_0 + \Delta t \Delta_U$

This can be related to the Taylor expansion of $U$ around $t_0$ :

$U_1 = U_0 + \Delta t \left[ f(U_0) + \text{high order terms} \right]$

with $\Delta_U \simeq f(U_0) + \text{high order terms}$.