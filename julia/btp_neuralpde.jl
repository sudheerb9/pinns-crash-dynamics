import Pkg;
Pkg.add("NeuralPDE")
using NeuralPDE, Flux, ModelingToolkit, GalacticOptim, Optim, DiffEqFlux
import ModelingToolkit: Interval, infimum, supremum

@parameters t
@variables x(..), y(..)

Dtt = Differential(t)^2
Dt = Differential(t)

# ODE
eq = [m1*Dtt(x)+ (c1+c2)*Dt(x) + (k1+k2)*x - c2*Dt(y) -k2*y, 
      m2*Dtt(y) + c2*Dt(y) - c1*Dt(x) + k2*y - k2*x]

# Initial and boundary conditions
bcs = [x(0.) ~ 0.0,
       y(0.) ~ 0.0]

# Space and time domains
domains = [t ∈ Interval(0.0,0.3)]

# Neural network
chain = FastChain(FastDense(2,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,20,tanh),FastDense(20,1))

discretization = PhysicsInformedNN(chain, QuasiRandomTraining(20))
@named pde_system = PDESystem(eq,bcs,domains,[t],[x(t),y(t)])
prob = discretize(pde_system,discretization)

cb = function (p,l)
    println("Current loss is: $l")
    return false
end

res = GalacticOptim.solve(prob, ADAM(0.01); cb = cb, maxiters=2000)
phi = discretization.phi


import Pkg;
Pkg.add("Plots")
using Plots

dx = 0.05
xs = [infimum(d.domain):dx/10:supremum(d.domain) for d in domains][1]
u_predict  = [first(phi(x,res.minimizer)) for x in xs]

x_plot = collect(xs)
plot(x_plot ,u_predict,title = "predict")
