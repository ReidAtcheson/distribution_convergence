import util
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.title("CG convergence w/ different eig distributions. cond(A)=1e6")

xmin=1e-6
xmax=1.0
m=10000
res=util.cg(util.sample_linear_pdf(xmin=xmin,xmax=xmax,size=m))
plt.semilogy(res,linewidth=2,label="Linearly decaying dist")

res=util.cg(util.sample_uniform(xmin=xmin,xmax=xmax,size=m))
plt.semilogy(res,linewidth=2,label="Uniform distribution")

res=util.cg(util.chebyshev_nodes(m,xmin,xmax))
plt.semilogy(res,linewidth=2,label="Chebyshev nodes")




plt.xlabel("CG iteration")
plt.ylabel("relative residual")
plt.legend()



plt.savefig("res.svg")
