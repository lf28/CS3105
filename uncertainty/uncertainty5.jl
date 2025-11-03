### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
begin
	using PlutoTeachingTools
	# using PlutoUI
	using Distributions, LinearAlgebra
	# using StatsPlots
	using LogExpFunctions
	# using SpecialFunctions
	# using Statistics
	# using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	# using Dagitty
	using PlutoUI

	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ b26dd8ed-bae2-4454-af53-31711fad8605
begin
	# using TikzGraphs
	# using Graphs
	using TikzPictures # this is required for saving
end;

# ╔═╡ 56013389-2242-4f23-b272-558fcdecd3b1
begin
	import TikzGraphs
	using Graphs
end

# ╔═╡ cf07faa6-e571-4c79-9dd2-f03041b99706
TableOfContents()

# ╔═╡ 3f49be0c-84d6-4bb1-adff-ab7f7ce15d05
ChooseDisplayMode()

# ╔═╡ cb9278ae-fd3f-4a2d-b188-ab463006db38
md"""


# CS3105 Artificial Intelligence


### Uncertainty 5
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 35494406-218b-41cd-b3af-6ba07d45d935
md"""

# Probabilistic inference algorithm
\

### Exact inference algorithm
"""

# ╔═╡ 9179c13b-1cc4-4ec6-91f2-aaed6a312ee9
md"""

## Recep: Bayes' net

"""

# ╔═╡ e67e822d-25c6-44a7-8ae2-13f42534d70b
TwoColumnWideRight(md"""
#### Bayes' net 

$\Large \text{bn} = \{G, \{P\}\}$

* ##### ``G``: _a DAG_
 


* ##### CPTs: for each node ``\large P(X_i|\text{parent}(X_i))`` 


""", html""" <center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglarCPTs.png" width = "700"/></center>
""")

# ╔═╡ eac84f66-a017-436e-b3a0-30a32677179e
md"""

## Recap: factoring property

"""

# ╔═╡ ed09a8e6-6b8f-415d-84f2-bcaed9d32558
md"""

!!! important "Factoring property"
	$\Large P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$
	
"""

# ╔═╡ 4f58814a-e990-445c-b959-f92d9aac1fe9
TwoColumnWideRight(md"""

\


#### Example

```math
\begin{align}
P&(B,E,A,J,M) = \\
&P(B)P(E) P(A|B, E)P(J|A)P(M|A)

\end{align}
```

""", html""" <center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglarCPTs.png" width = "700"/></center>
""")

# ╔═╡ 1769f7b4-4212-487a-bad3-bfaa3bf7026c
md"""

## Probabilistic inference 

!!! infor "Probabilistic Inference"
	```math
	\Large P(Q|E_1=e_1, E_2= e_2, \ldots, E_k= e_k)
	```
	\

	* #### ``Q``: **query** random variable
	* #### ``E_i``: **evidence** or _observed_ (evidence can be empty) 
    * #### ``N_i``: all the others: **nuisance** random variables 
"""

# ╔═╡ 89be5e5a-76e6-446a-aa7a-00d0cccd0c33
md"""

## Some inference examples



"""

# ╔═╡ 5d45e5db-6b95-48c0-8471-c062c38aab2b
TwoColumnWideLeft(

md"""

#### ``P(J)``: _how likely John calls ?_
* ##### query: ``J``
* ##### evidence: ``\emptyset``
* ##### nuisance: ``B,E,A,M``


#### ``P(J|+b)``: _how likely John calls if Burglary happens?_
* ##### query: ``J``
* ##### evidence: ``B=+b``
* ##### nuisance: ``E, A`` and ``M``



#### ``P(B|+j,+m)``: _how likely Burglary happens given both John and Mary calls?_
* ##### query: ``B``
* ##### evidence: ``J=+j, M =+m``
* ##### nuisance: ``E`` and ``A``

""",

	
html"""<br><br><br><br><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "300"/></center>"""
	
)

# ╔═╡ 4edc91cb-89d6-45df-8055-c4752189bf7a
# md"""


# ## Query types


# * **bottom-up**: given evidence (observations, data) infer the cause; opposite the direction of edges, 
#   * *e.g.* $P(B|J,M)$
# * **top-down**: given cause infer downstream r.v.s (follow the direction of edge), 
#   * prediction of future data
#   * *e.g.*  $P(J|B, E)$

# * **mixture of both**: $P(J|M)$

# """

# ╔═╡ 32770f48-6f30-435a-86b2-b1ad272b29ff
# html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "300"/></center>"""

# ╔═╡ 7518fc3a-685c-4089-9be8-1edb332a68d2
md"""

## The inference formula


!!! information ""
	#### Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\Large \begin{align}
	P(Q|\mathcal{E}=\mathbf{e}) &\propto P(Q, \mathcal{E}=\mathbf{e}) \\
	&= \sum_{n_1}\ldots \sum_{n_k} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)\\
	&= \sum_{n_1}\ldots \sum_{n_k} \left ( \prod_i \text{CPFs}\right )
	\end{align}


	```
	* #### ``Q``: query random variable 
	* #### ``\mathcal{E}``: conditioned evidence 
	* #### ``\mathcal{N}``: nuisance (or hidden) random variables

"""

# ╔═╡ 8a0f33b2-28d1-475e-a30f-289fec3b59bd
md"""

## An example 

#### Let's consider 


$\Large P(J|+b, -e)$


!!! question ""
	#### *By conditional probability definition*
	```math
	\large P(J|+b, -e) = \frac{P(J, +b, -e)}{P(+b, -e)} = \frac{P(J, +b, -e)}{P(+j, +b, -e) + P(-j, +b, -e)}
	```

#### Note that the denominator (sum rule) is a **normalising constant**:

```math
\large P(+b, -e) = P(+j, +b, -e) + P(-j, +b, -e)\tag{sum rule}
```

*  ##### and it can be computed from the numerator ``P(J, +b, -e)`` 


##

!!! question ""
	#### Therefore, as a short-hand notation, queries are often written as  

	$$\Large P(J|+b, -e) \propto P(J, +b, -e)$$ 

	#### or 
	
	$$\Large P(J|+b, -e) =\alpha P(J, +b, -e)$$ 
	* ##### where ``\alpha = \frac{1}{P(+b, -e)}`` is the normalising constant's inverse


"""

# ╔═╡ 96c86194-fe8a-444d-ae43-fc98ad6e5181
md"""
## An example (cont.)

#### Next, by ``\textcolor{red}{\text{sum rule}}`` 
* ##### sum out the _nuisance_ random variables

!!! question ""
	$$\large\begin{align}
	P(J|+b, -e) &= \alpha P(J, +b, -e)\\
	\Aboxed{&= \alpha \sum_{a'}\sum_{m'} P(+b,-e,a',J,m')\;\;\;\textcolor{red}{\text{sum rule}}}
	\end{align}$$



"""

# ╔═╡ 22134218-1606-45b0-89e2-f199bb17b15b
md"""
## An example (cont.)

#### Next, by ``\textcolor{blue}{\text{factoring property}}`` of the BN

!!! question ""
	$$\large \begin{align}
	&P(J|+b, -e) = \alpha P(J, +b, -e)\\
	&= \alpha \sum_{a'}\sum_{m'} P(+b,-e,a',J,m')\;\;\;\textcolor{red}{\text{sum rule}}\\
	\Aboxed{&=\alpha \sum_{a'}\sum_{m'} P(+b)P(-e)P(a'|+b,-e)P(J|a')P(m'|a')\;\;\;\textcolor{blue}{\normalsize\text{factor property}}} 
	\end{align}$$

"""

# ╔═╡ 8c2f8f52-c440-4ad3-a08e-290156552c08
md"""


## Further details

#### As a concrete example, to find, *e.g.*

$\large P(J|+b, -e)$


#### For $J=+j$ : 

$$\begin{align}
P(J&=+j|+b,-e) = \alpha \sum_{a'}\sum_{m'}P(+b)P(-e)P(a'|+b,-e)P( +j|a')P(m'|a')\\
&= \alpha\times (+)\begin{cases}
P(+b)P(-e)P(-a|+b,-e)P(+j|-a)P(-m |-a) & a'=-, m'=- \\
P(+b)P(-e)P(-a |+b,-e)P(+j|-a )P(+m|-a ) & a'=-, m'=+ \\
P(+b)P(-e)P(+a|+b,-e)P(+j|+a)P(-m|+a)& a'=+, m'=- \\
P(+b)P(-e)P(+a|+b,-e)P(+j|+a)P(+m|+a) & a'=+, m'=+ 
\end{cases} \\
&= \alpha\times (+)\begin{cases}
0.001 \times (1-0.002) \times (1-0.94) \times 0.05 \times (1-0.01) & a'=-, m'=- \\
0.001 \times (1-0.002) \times (1-0.94) \times 0.05 \times 0.01 & a'=-, m'=+ \\
0.001 \times (1-0.002) \times 0.94 \times 0.9 \times (1-0.7) & a'=+, m'=- \\
0.001 \times (1-0.002) \times 0.94 \times 0.9 \times 0.7 & a'=+, m'=+ 
\end{cases} \\
&=\alpha \times 0.00847302
\end{align}$$

#### For $J=-j$ : 

$$\begin{align}
P(J&= -j|+b, -e) = \alpha \sum_{a'}\sum_{m'}P(+b)P(-e)P(a'|+b,-e)P(-j|a')P(m'|a')\\
&= \alpha\times (+)\begin{cases}
\ldots & a'=-, m'=- \\
\ldots & a'=-, m'=+ \\
\ldots & a'=+, m'=- \\
\ldots & a'=+, m'=+
\end{cases} \\
&= \alpha \times 0.0001507

\end{align}$$

#### *Lastly*, normalise to find 
$\alpha = \frac{1}{0.0008473+ 0.0001507}$

$$\begin{align}
P(J|+b, -e) &\propto \begin{cases} 0.00847302 & J= +j \\ 0.0001507 & J=-j \end{cases}\\
&= \begin{cases} 0.849 & J= +j \\ 0.151 & J=-j \end{cases}
\end{align}$$


"""

# ╔═╡ bbbd12d6-ec08-482c-a82b-e20161ebd507
# md"""

# !!! question "Question"
# 	##### How many _multiplications_ involved?
# """

# ╔═╡ 2ac736f1-b27c-46a4-8ae4-d929a2efa5d8
# Foldable("", md"
# Roughly in the order

# ```math
# \Large 2^{N_k} \times \text{\# of factors} 
# ```

# * ##### ``N_k``: the number of nuisance variables
# * ##### ``\#`` of factors: or number of nodes in a BN
# * ##### ``O(2^{N_k}\cdot N)`` complexity
# ")

# ╔═╡ c7a8c934-e45d-4f34-9429-b7947cea95f8
md"""

## Another example: ``P(B|J=j, M=m)``

$\Large P(B|J=j, M=m)$
* ##### the nuisance random variables: ``\{E,A\}``

\

$$\large \begin{align}&P(B|J=j,M=m) = \alpha P(B,J=j,M=m) \;\;\;\;\;\;\text{\normalsize:conditional probability} \\
&= \alpha \sum_{e'}\sum_{a'} P(B,e',a',J=j,M=m)\;\;\;\;\;\;\;\;\;\;\;\;\;\,\;\;\;\;\;\;\;\;\;\; \normalsize\text{:sum rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B)P(e')P(a'|B,e')P(J=j|a')P(M=m|a') \;\; \text{\normalsize:factor property}\end{align}$$

\

* ##### for each ``B=b``, `enumerate` all ``(e', a') \in \{\texttt{t}, \texttt{f}\}^2`` and `sum`
* ##### then `normalise`
"""

# ╔═╡ 0ab4585d-9d91-4457-a112-6df073c1f4c0
# md"""
# ## Mixture of both ``P(J|M=m)``



# ###### For query ``P(J|M=m)``, nuisance random variables: $\{B,E, A\}$

# $$
# \begin{align}
# P(J|M&=m) = \alpha P(J, M=m)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;\;\, \;\;\;\;\;\; \;\;\;\;\;\,\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;  \color{blue}\text{:conditional probability}\\
# &= \alpha \sum_{b'}\sum_{e'}\sum_{a'} P(B=b',E=e',A=a',J,M=m)\;\;\, \;\;\;\;\;\;\;\color{blue}\text{:sum rule}\\
# &=\alpha \sum_{b'}\sum_{e'}\sum_{a'}  P(b')P(e')P(a'|b',e')P(J|a')P(M=m|a')\,\color{blue}\text{:factoring property}
# \end{align}$$

# Specifically,

# * for each ``J=j``, `enumerate` ``(b', e', a')\in \{\texttt{t,f}\}^3``, then `sum`, 
# * then `normalise`
	
# """

# ╔═╡ e2a88805-3f01-4ab3-8b9e-a3e4012b49b7
md"""

## The formula


!!! information ""
	#### Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\large \begin{align}
	P(Q|\mathcal{E}=\mathbf{e}) &\propto P(Q, \mathcal{E}=\mathbf{e}) \\
	&= \sum_{n_1}\ldots \sum_{n_k} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)\\
	&= \sum_{n_1}\ldots \sum_{n_k} \left ( \prod_i \text{CPFs}\right )
	\end{align}


	```
	* #### ``Q``: query random variable 
	* #### ``\mathcal{E}``: conditioned evidence 
	* #### ``\mathcal{N}``: nuisance (or hidden) random variables

\


#### More nuisance random variables ``\Rightarrow`` more sums



"""

# ╔═╡ 3131731b-f6de-43d5-bf78-8b1f321a04c5
aside(tip(md"""

``\mathcal{N, Q, E}``: this font denotes sets of random variables

"""))

# ╔═╡ 84481277-4b3f-4c9c-9181-a32e76fc8546
md"""

## A small improvement 


#### We can improve the algorithm a little bit

* ##### reducing some floating number multiplications


* ##### based on distribution law


$\large ax + ay = a(x+y)$
"""

# ╔═╡ 16c3e05c-35f1-4857-bbb4-e0feed7507fd
md"""
## Digress: summation notation

#### Recall the summation  

$\Large \sum_{i=1}^n ax_i =ax_1 + a x_2 + \ldots + ax_n= a\left (\sum_{i=1}^n x_i\right )$

* ##### when ``a`` is a common factor, 
  * it can be taken out or equivalently pushing summation $\,\Sigma_{\cdot}\,$ inwards
\

* ##### it saves some floating multiplication operations (why)
  * remember floating number ``\times`` is more expensive than ``+/-``



##

#### Example: ``P(J|+b, -e)``




"""

# ╔═╡ 514f8322-f2f4-4384-8661-3c7683d21a78
md"""


$$\begin{align}
P(J&=+j|+b,-e) = \alpha \sum_{a'}\sum_{m'}\boxed{P(+b)P(-e)}P(a'|+b,-e)P( +j|a')P(m'|a')\\
&= \alpha\times (+)\begin{cases}
\boxed{P(+b)P(-e)}P(-a|+b,-e)P(+j|-a)P(-m |-a) & a'=-, m'=- \\
\boxed{P(+b)P(-e)}P(-a |+b,-e)P(+j|-a )P(+m|-a ) & a'=-, m'=+ \\
\boxed{P(+b)P(-e)}P(+a|+b,-e)P(+j|+a)P(-m|+a)& a'=+, m'=- \\
\boxed{P(+b)P(-e)}P(+a|+b,-e)P(+j|+a)P(+m|+a) & a'=+, m'=+ 
\end{cases} \\
&= \alpha\times (+)\begin{cases}
\boxed{0.001 \times (1-0.002)} \times (1-0.94) \times 0.05 \times (1-0.01) \\
\boxed{0.001 \times (1-0.002)} \times (1-0.94) \times 0.05 \times 0.01  \\
\boxed{0.001 \times (1-0.002)} \times 0.94 \times 0.9 \times (1-0.7)  \\
\boxed{0.001 \times (1-0.002)} \times 0.94 \times 0.9 \times 0.7 
\end{cases} \\
&=\alpha\times (+)\left (\boxed{0.001 \times (1-0.002)} \times\begin{cases}
  (1-0.94) \times 0.05 \times (1-0.01)  \\
 (1-0.94) \times 0.05 \times 0.01  \\
 0.94 \times 0.9 \times (1-0.7)  \\
0.94 \times 0.9 \times 0.7 
\end{cases}\right)
\end{align}$$

"""

# ╔═╡ 469b80f9-26db-4d16-9980-7c7e100ba486
md"""

##
#### That is 
$$\large \begin{align}

P(J|+b,-e)&= \alpha \sum_{a'}\sum_{m'} \underbrace{\boxed{P(+b)P(-e)}}_{\text{constant!}}P(A=a'|b,e)P(J|a')P(m'|a') \\
  &= \alpha P(+b)P(-e)\sum_{a'}\sum_{m'}{P(a'|b,e)P(J|a')}P(m'|a')
\end{align}$$



* ##### ``P(+b)P(-e)`` is a common factor




"""

# ╔═╡ 86c2dd5d-d7c6-4f4b-a793-06aacc15375c
md"""
## Example (conti.)


$$\large \begin{align}
P(J|B=b,E=e)
  &= \alpha P(b)P(e)\sum_{a'}\sum_{m'}\boxed{P(a'|b,e)P(J|a')}P(m'|a')
\end{align}$$

#### ``\boxed{P(a'|b,e)P(J|a')}`` is a common factor (from $m$'s perspective)
* ##### or in other words, we push $\sum_{m'}$ inwards

$$\large\begin{align}
P(J|B=b,E=e)
  = \alpha P(b)P(e)\sum_{a'}\boxed{P(a'|b,e)P(J|a')}\sum_{m'}P(m'|a')\end{align}$$

"""

# ╔═╡ 28611165-f29b-4163-bf02-796598804129
# md"""
# ## Further simplification

# We can further simplify it, note that 


# $$\sum_{m'}P(M=m'|a)=1$$

# $$\Rightarrow P(J|b,e)
#   = \alpha P(b)P(e)\sum_{a'}P(A=a'|b,e)P(J|A=a')\underbrace{\sum_{m'}P(M=m'|a')}_{1.0}$$

# So we have 


# $$P(J|b,e)
#   = \alpha P(b)P(e)\sum_{a'}P(A=a'|b,e)P(J|A=a')$$

# * 2 nested sums to 1 summation

# * in general, it reduces from ``O(N \times 2^N)`` to ``O(2^N)``
# * the difference is we compute while we back-track rather than wait until the end
# """

# ╔═╡ 42e93176-c83d-49af-91b8-f01cda668edd
# md"""
# ## More example

# To find, *e.g.*

# $P(J|B=\texttt t, E=\texttt f);$


# For $J=\texttt t$: 

# $$\begin{align}
# P(J=\texttt t|+b,-e) &= \alpha P(+b)P(-e)\sum_{a'}P(A=a'|+b,-e)P(J=\texttt t|A=a')\\
# &= \alpha\times .001 \times (1-.002) \times (.94\times .9 + (1-.94)\times.05) \\
# &= \alpha 0.0008473
# \end{align}$$

# For $J=\texttt f$: 

# $$\begin{align}
# P(J=\texttt f|+b,-e) &= \alpha P(+b)P(-e)\sum_{a'}P(A=a|+b,-e)P(J=\texttt f|A=a')\\
# &= \alpha\times .001 \times (1-.002) \times (.94\times (1-.9) + (1-.94)\times(1-.05))\\
# &= \alpha 0.0001507

# \end{align}$$

# Normalise to find 

# $\alpha = \frac{1}{0.0008473+ 0.0001507}$
# $$P(J|B=\texttt t, E=\texttt f) = \begin{cases} 0.849 & J=\texttt t \\ 0.151 & J=\texttt f \end{cases}$$


# """

# ╔═╡ 693e0834-a970-44e4-a273-b71233984d70
md"""

## Another example



$$\large \begin{align}P&(B|j,m) = \alpha P(B,j,m)\;\;\; \text{conditional probability rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B,e',a',j,m)\;\;\; \text{summation rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B)P(e')P(a'|B,e')P(j|a')P(m|a') \;\;\; \text{factoring CPTs} \\
&= \alpha P(B)\sum_{e'}P(e')\sum_{a'} P(a'|B,e')P(j|a')P(m|a')\;\;\; \text{simplify} \end{align}$$


"""

# ╔═╡ bc2b6d66-ceb0-4fcb-8af7-53c3dde09f20
md"""

## How to implement the inference algorithm



#### So far, we have computed by hand
\

#### _How to implement it_?

"""

# ╔═╡ 27d8fc53-e121-4e3b-9861-77fcbaf00cd3
md"""


## A key step: enumerate the nuisance 



!!! information ""
	#### Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\large \begin{align}
	P(Q|\mathcal{E}
	&= \sum_{n_1}\ldots \sum_{n_k} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)\\
	&= \sum_{n_1}\ldots \sum_{n_k} \left ( \prod_i \text{CPFs}\right )
	\end{align}


	```
	* #### ``\mathcal{N}``: nuisance (or hidden) random variables



## Digress: enumerate all - how ?


#### To enumerate ``(x_1, x_2) \in (\texttt{t}, \texttt{f})^2``
* ##### a nested `for` loop works

```julia
for x₁ in X₁ # assume X₁ returns the possible value x₁ can take
	for x₂ in X₂
		print(x₁, x₂) # or compute something interesting
	end
end
```


#### For three ``X_1, X_2, X_3``, 
* ##### obviously, nest 3 `for` loops

```julia
for x₁ in X₁ # assume X₁ returns the possible value x₁ can take
	for x₂ in X₂
		for x₃ in X₃
			print(x₁, x₂, x₃) # or compute something interesting
		end
	end
end
```

## How to enumerate a dynamic list of vars

!!! question "Question"
	#### What if the list ``\{X_1, X_2, \ldots, X_n\}``'s size is not known before-hand
    * #####  ``n`` is not known or fixed
    * ##### how to enumerate a **dynamic** list


"""

# ╔═╡ 74125061-4bd4-41d5-aeeb-436074db9ded
# md"""

# ## A naive inference algorithm



# ------
# **A naive exact inference algorithm**

# **Step 0.** tabulate and store the full joint distribution based on 

# ```math
# P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))
# ```


# **Step 1.** for each value ``q`` that ``Q`` can take (*i.e.* apply sum rule)

#    * find rows matches ``Q=q \wedge \mathcal{E} =\mathbf e``
#    * sum their probabilities and store in ``\texttt{vec}[q]``


# **Step 2.** normalise ``\texttt{vec}`` then return it
  
# $$P(Q=q|E=\mathbf{e}) = \frac{\texttt{vec}[q]}{\sum_q \texttt{vec}[q]}$$

# ------	
# """

# ╔═╡ 9c58d982-b26a-4d98-9d04-c07ee3816e4a
# md"""
# ## A naive inference algorithm

# For example, query 

# $P(J|B=\texttt{t},E=\texttt{t})$

# **Step 0**: populating the full joint table:


# """

# ╔═╡ 9f49f45d-e350-480e-a2c0-2fb48ada6d88
# html"
# <table>
# <thead>
#     <tr>
#         <td>B</td>
#         <td>E</td>
#         <td>A</td>
#         <td>J</td>
#         <td>M</td>
# 		<td>P(B,E,A,J,M)</td>
#     </tr>
#   </thead>
#     <tr>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
# 		<td>..</td>
#     </tr>
# 	<tr>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
#         <td>t</td>
# 		<td>..</td>
#     </tr>

# 	<tr>
#         <td>f</td>
#         <td>f</td>
#         <td>f</td>
#         <td>t</td>
#         <td>f</td>
# 		<td>..</td>
#     </tr>

# 	<tr>
#         <td>f</td>
#         <td>f</td>
#         <td>t</td>
#         <td>t</td>
#         <td>t</td>
# <td>..</td>
#     </tr>

# 	<tr>
#         <td>f</td>
#         <td>t</td>
#         <td>t</td>
#         <td>f</td>
#         <td>f</td>
# <td>..</td>
#     </tr>

# 	<tr>
# 		<td> ⋮</td> 
#         <td>⋮</td>
#         <td>⋮</td>
#         <td>⋮</td>
#         <td>⋮</td>
#         <td>⋮</td>
#     </tr>

# 	<tr>
#         <td>t</td>
#         <td>t</td>
#         <td>t</td>
#         <td>t</td>
#         <td>t</td>
# <td>..</td>
#     </tr>
# </table>"

# ╔═╡ 2a7462da-7a1a-4ab7-8544-5e2be106fd43
# md"""

# **Step 1**: enumerate query variable ``J`` and apply sum rule

# * for ``J=\texttt{t}``, `filter` the table with the condition ``\{J=\texttt{t} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
#   * sum and store it in ``\texttt{vec}[J=\texttt{t}]``
# * for ``J=\texttt{f}``, `filter` the table with the condition ``\{J=\texttt{f} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
#   * sum and store it in ``\texttt{vec}[J=\texttt{f}]``

# **Step 2** normalise the ``\texttt{vec}`` and return it
# """

# ╔═╡ dfd22bfb-ff0a-4847-8cb1-24ef1d24f3c9
# (tip(md"""

# What **step 1** essentially does (sum out the nuisances):

# $$\begin{align}&\texttt{vec}[J=t] =\sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=t, M=m')\end{align}$$


# $$\begin{align}&\texttt{vec}[J=f] = \sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=f, M=m')\end{align}$$

# """))

# ╔═╡ ce48b420-bb2b-4a85-a875-2084897b0134
# md"""

# ## Naive algorithm does not scale
# """

# ╔═╡ ac943f36-669e-4799-8b31-6c8d01c328ba
# md"""

# ##### *Step 0* is not feasible !
# \


# ###### The joint full table is **HUGE** 

# * ``2^N`` rows, ``N`` is the number of random variables in the BN
# * space complexity is ``O(2^{N})``


# The rest two steps, however, are plainly _simple and efficient_
# * just basic database `filter` operation and summation
# """

# ╔═╡ 8e330eb7-7790-438e-bce6-9c02b45a8dc9
md"""

## Digress: enumerate all - Backtracking



#### Recursive backtracking! 
* ##### or recursive DFS tree search



"""

# ╔═╡ ee50d06e-eb24-49f1-9f56-846b097a3238
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/back_track.png" width = "650"/></center>"""

# ╔═╡ 75cb72a8-7620-4b63-9a48-85414141354c
md"""
## 
#### The **inputs**:

* `cur_idx` = 1,2, ..., `length(Xs)`: current index of the variable to enumerate;
* `Xs`: a list of variables to be enumerated, *e.g.* `Xs = [[0,1], [0,1], ...]`
* `storage = [_, _, ..., _]`: a pre-allocated fix-sized array to rewrite the combinations

```julia
function enum_all(cur_idx, Xs, storage = [...])
	# base case, when the storage is full
	if cur_idx > length(Xs)
		println(tape) # or do some computation
	else
		for each x in Xs[cur_idx]
			# fill the tape at cur_idx
			storage[cur_idx] = x
			# move on to the next variable recursively
			enum_all(cur_idx + 1, Xs, storage)
		end
	end
end


```

"""

# ╔═╡ 0406ae12-8be2-478f-993f-10837abfcadf
md"""

## Demonstration
"""

# ╔═╡ 67a909b7-ca69-4d0d-9a74-1d0622fb8397
md"Show backtrack tree: $(@bind show_tree CheckBox(default=false))"

# ╔═╡ 5d8f7bfc-7192-4ce4-a523-15abc80047dd
function enum_all(Xs, cur_idx, storage)
	if cur_idx > length(Xs)
		println(storage)	# or compute something interesting with the combination
	else
		for v in Xs[cur_idx]
			storage[cur_idx] = v
			enum_all(Xs, cur_idx+1, storage)
		end
	end
end

# ╔═╡ 2a2437d6-b774-42f8-887e-6b5d4940b367
begin
	num_of_vars = 4
	num_of_choices = 0:1
	Xs = [num_of_choices for _ in 1:num_of_vars]
	# Xs = [0:1, 1:3]
end;

# ╔═╡ e13c61c8-f0a8-49be-a523-9aca895640f2
enum_all(Xs, 1, Array{Int}(undef, length(Xs)))

# ╔═╡ 47e5143b-b7d5-4ba9-a0fa-b1abd74b1bde
let
	function enum_all_with_tree!(Xs, cur_idx, storage, graph = [], names = [])
		if cur_idx > length(Xs)
			return
		else
			for v in Xs[cur_idx]
				storage[cur_idx] = v
				# store the edges between the recursive function call
				push!(graph, storage[1:cur_idx-1] => storage[1:cur_idx])
				push!(names, storage[1:cur_idx]) # the names' storages in DFS order
				enum_all_with_tree!(Xs, cur_idx+1, storage, graph, names)
			end
		end
	end;
	if show_tree
		graph = []
		names = [[]]
		enum_all_with_tree!(Xs, 1, Array{Int}(undef, length(Xs)), graph, names)
		name_dict = Dict()
		names_strings = String[]
		for (i, n) in enumerate(names)
			push!(name_dict, n=>i)
			# some boring but tedious processing to make the graph look tidy
			st_str = ["\\_" for _ in 1:length(Xs)]
			[st_str[j] = string(v) for (j, v) in enumerate(n)]
			st_str_ = st_str[1]
			[st_str_ *= " "*string(x)   for x in st_str[2:end]]
			push!(names_strings, st_str_)
		end
		g = DiGraph(length(names))
		for (s, d) in graph
			si = name_dict[s]
			di = name_dict[d]
			add_edge!(g, si=> di)
		end
		TikzGraphs.plot(g, names_strings, options="scale=2, font=\\large\\sf")
	end
end

# ╔═╡ 4a4a91c0-5c25-4e54-99e1-c731c53f4972
# md"""
# ## A second attempt

# ###### Form the full joint table ``P(X_1, X_2, \ldots, X_n)`` in one-go is *neither*
# * **necessary**
# * nor **practical**


# ###### The idea of second attempt:
# * use recursive *back-tracking* to *enumerate* the **nuisance random variables** only ``\mathcal{N}=\{N_1, N_2, \ldots N_k\}``

# -----

# **Step 0.** initialise an emtpy array: ``\texttt{vec}``

# **Step 1.** for each value ``q`` that ``Q`` can take
#   * recursively enumerate all combinations ``(n_1, n_2,\ldots, n_k)`` by backtrack and **accumulates**
# $$\texttt{vec}[q]  \mathrel{+}=  P(Q=q, E=\mathbf e, N_1 = n_1,  \ldots, N_k=n_k)$$

# **Step 2.** normalise ``\texttt{vec}`` then return it
  
# $$P(Q=q|E=\mathbf{e}) = \frac{\texttt{vec}[q]}{\sum_q \texttt{vec}[q]}$$

# ------	
# """

# ╔═╡ 98ebe05d-dd1f-4b6b-afab-e7f42606d5bb
# md"""

# ## A second attempt *vs* the first attempt


# ##### The second attempt is better 

# * we do not enumerate all but a smaller subset


# Recall recursive backtrack essentially builds a tree

# * the first attempt creates a full tree and **stores** it somewhere in RAM

# * the second method creates a much smaller sub-tree and **does not** store them

# ##

# ##### However, the worst case is still very bad

# Consider the extreme case in which ``\mathcal{E} =\emptyset``, *i.e.* no evidence

# ```math
# P(Q) \propto \underbrace{\sum_{n_1}\sum_{n_2}\ldots \sum_{n_k}}_{\texttt{N-1 of nuisance r.v.s} } P(Q, N_1 = n_1, \ldots, N_k =n_k)
# ```
# * space complexity is fine (backtracking is efficient in space complexity, you cannot beat it)
#   * we do not store the table
# * but time complexity is ``O(N \times 2^N)``
#   * ``2^N`` combinations, each has ``N`` floating number multiplications
# """

# ╔═╡ 75c4a259-7d39-49bf-bda8-1e5a89770598
# using Graphs

# ╔═╡ a4d35349-ac25-4e47-adfc-c31e6798e265
md"""

## Enumerate-ask algorithm

#### `ENUM_ASK`: for each ``q'``, compute unnormalised ``p(q', \mathbf{e})``, then normalise
-----
**function** ``\texttt{ENUM\_ASK}``(``{Q}``, ``\{\mathcal{E}=\mathbf{e}\}``, ``\texttt{bn}``)

``\;\;`` ``\texttt{Xs} = \texttt{bn.VARS}\;\;\;\;\;``   # *i.e.* 
 list of all r.v.s in a topological order (parents before child nodes)

``\;\;`` ``\texttt{vec} \leftarrow \texttt{an array with all zeros}\;\;\;\;\;\;`` # an array for ``P(q|\mathbf e)``

``\;\;`` **for each**  value ``q`` of ``Q`` **do**

  * ``\texttt{vec}(q) \leftarrow \texttt{ENUM-ALL}(\texttt{Xs}, 1, \{\mathcal{E}=\mathbf{e}\} \cup \{Q=q\})``
``\;\;`` **endfor**

``\;\;`` **return** ``\texttt{NORMALISE}(\texttt{vec})``

**end**

-----

\


#### `ENUM_ALL`: the backtracking bit
-----
**function** ``\texttt{ENUM\_ALL}````(````\texttt{Xs}``, ``\texttt{idx}``, ``\{\mathcal{E}=\mathbf{e}\}````)``

``\;\;`` **if** ``\texttt{idx} > \texttt{length}(\texttt{Xs})`` **then return** ``1.0``

``\;\;`` ``{X} \leftarrow \texttt{Xs}[\texttt{idx}]``

``\;\;`` **if** ``X \in \{\mathcal{E}=\mathbf{e}\}`` # if the variable is in the evidence set or has been assigned

``\;\;\;\;\;\;`` **return** ``P({x}| \texttt{parents}(X))\times \texttt{ENUM-ALL}(\texttt{Xs},  \texttt{idx+1}, \{\mathcal{E}=\mathbf{e}\})``

``\;\;`` **else**  ``\;\;`` # X is nuisance, apply sum rule 



``\;\;\;\;\;\;`` **return** ``\sum_x P(x |\texttt{parents}(X)) \times \texttt{ENUM-ALL}(\texttt{Xs}, \texttt{idx+1}, \{\mathcal{E}=\mathbf{e}\} \cup \{X =x\})``

``\;\;`` **end**

**end**

-----


* ##### note that Xs should be in a topological order (parents before child)



"""

# ╔═╡ a9d05d1b-f789-4a7f-80bf-767cd0981331
md"""

## An example
"""

# ╔═╡ 2cd4b610-2fbc-43ef-8c64-1480e60356ba
md"""

$$\Large\begin{align}P(b|j,m)
&\propto P(b)\sum_{e'}P(e')\sum_{a'} P(a'|b,e')P(j|a')P(m|a')\end{align}$$

"""

# ╔═╡ 430cc0b0-4748-4900-be65-22c1f8b0ec02
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/enum_ask.png" width = "450"/></center>"""

# ╔═╡ d0ea7686-9d39-40d3-a695-8c71783a4a1f
md"""
## Summary: exact inference


#### `ENUM-ASK` algorithm is an exact inference algorithm

* ##### Time complexity: ``O(2^N)`` (``N`` is the size of the BN)



* ##### A *rule of thumb*: for a BN with 25+ nodes, exact inference no longer works
  * ##### we need approximate inference algorithms

"""

# ╔═╡ 748ade25-bb47-4a67-a624-663fb7a5933f
md"""

# Why `enum-all` at all?
"""

# ╔═╡ 344a3078-fad8-4985-8270-c52b7be034f2
md"""


## Why bother _summing out_ the nuisances ?


### _esp._ if it is slow
!!! information ""
	```math
	\Large\begin{align}
	 P(Q|\mathcal{E}=\mathbf{e}) &\propto P(Q, \mathcal{E}=\mathbf{e}) \\
	&= \Large \colorbox{lightpink}{$\sum_{n_1}\ldots \sum_{n_k}$} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)
	\end{align}
	```
	\

    * #### more nuisance variables ``\Rightarrow`` more sums ``\Rightarrow`` slow and intractable


"""

# ╔═╡ 409d6b4d-f0c3-44b0-bfab-7331f434de9b
begin
	g = DiGraph(3)
	add_edge!(g, 1, 2)
	add_edge!(g, 1, 3)
	# add_edge!(g, 1, 3)
	graphplt = TikzGraphs.plot(g, [L"\textit{Coin}", L"Y_1", L"Y_2"], options="scale=2, font=\\Huge", node_style="draw", graph_options="nodes={draw,circle}")
end;

# ╔═╡ 8bd9e5a8-4534-4c2b-b6e6-5736bd9ec745
md"""

## Coin guess example


!!! problem "Coin guess"
	#### Two coins in an urn
	* ##### Coin 1 fair: $p_{1}= 0.5$
	* ##### Coin 2 bent: $p_2=0.01$: very **likely** to observe a `tail`
	#### Randomly pick one coin 
	* ##### toss it once $Y_1$ and it is a `tail`
	* ##### predict the next toss $Y_2$

"""

# ╔═╡ b44bedea-a0d8-497c-9285-ab5f7aed6d60
md"""
## The Bayes' net
"""

# ╔═╡ 36c02a04-f925-4bfe-a530-b63d89b5e620
md"""

| |
| --- |
| $(graphplt)|
"""

# ╔═╡ f8a835e6-55f7-42f1-84c2-ee977d70dc38
md"""

## The query


#### To predict $Y_2$, the query is

```math
\Large
P(Y_2|Y_1 = \texttt{t})
```

* ##### the *nuisance* variable: ``C``

## But, why not do it `ad-hoc`?

#### Since we have observed ``Y_1=```tail`
* ##### it is more likely the bent coin ``C=2`` has been used

$$\Large P(C|Y_1= \texttt t) \approx \begin{cases}0.385 & C=1 \\ \Huge\boxed{0.615} & \Huge \boxed{C=2}\end{cases}$$ 



* ##### we could just use $C=2$ for future prediction, or treating ``\Large\boxed{C=2}`` as a gospel

```math
\Large
P(Y_2|Y_1=\texttt{t}) \approx ? P(Y_2|C=2)= \begin{cases} 0.01 & Y_2= \texttt{h} \\ \Huge\boxed{0.99} & \Huge\boxed{Y_2=\texttt{t}}\end{cases}
```


* ##### no enumerate and sum of the nuisance $C$! it is indeed fast

"""

# ╔═╡ 56acc259-7bc1-4f40-8745-b41950fba0ac
# md"""
# ## More concretely



# #### 1. one first finds the most likely coin choice ``C`` based on ``Y_1=\texttt{t}``:

# ```math
# \Large
# \hat C \leftarrow \arg\max_c P(C=c|Y_1=\texttt{t})
# ```


# * ##### ``\hat C = 2``, *i.e.* the second (bent) coin is most likely to have been used


# #### 2. then predict by treating ``\hat C=2`` as a gospel!

# ```math
# \Large
# P(Y_2|C=2)= \begin{cases} 0.01 & Y_2= \texttt{h} \\ 0.99 & Y_2=\texttt{t}\end{cases}
# ```


# """



# ╔═╡ 2de5daf9-b422-43fb-a22a-7c3c68ae0bf1
md"""

## Any problem with this approach?


!!! note "The problem with an optimisation based approach"
	
    
"""

# ╔═╡ 2502e421-e8a4-4712-9cb3-3ada7646edfb
Foldable("", md"""
#### The uncertainty associated with ``P(C|Y_1 =\texttt{t})`` is ignored! 
* ##### it is not yet conclusive that coin 2 is 100% used at all!
  * ##### about 40% change that coin 1 could have been used

$$\Large P(C|Y_1= \texttt t) \approx \begin{cases}\boxed{0.385 } & \boxed{C=1} \\ 0.615 & C=2\end{cases}$$ 
\




#### The prediction almost rules out the possibility of observing future `head` after only observing one `tail`


```math
\large
P(Y_2|Y_1=t) \approx ? P(Y_2|C=2)= \begin{cases} 0.01 & Y_2= \texttt{h} \\ \Huge\boxed{0.99} & \Huge\boxed{Y_2=\texttt{t}}\end{cases}
```
  * ##### this is the same as overfitting in machine learning 

""")

# ╔═╡ 9bbf617a-d355-48cd-a150-c276d9e61a85
md"""


## How to do better ?


### Sum-out the nuisance!



$$\Large \begin{align}P(Y_2|Y_1=y_1) &\propto P(Y_1=y_1, Y_2) \\
&= \sum_{c=1,2} P(C=c, Y_1 =y_1, Y_2)
\end{align}$$ 



"""

# ╔═╡ 42141d4d-3a11-4240-acb0-91bf412962f8
md"""


## How to do better ?


### Sum-out the nuisance!



$$\Large \begin{align}P(Y_2|Y_1=y_1) &\propto P(Y_1=y_1, Y_2) \\
&= \sum_{c=1,2} P(C=c, Y_1 =y_1, Y_2) \\
&= \sum_{c=1,2} P(C=c){P(y_1|C=c)}P(Y_2|C=c)
\end{align}$$ 



"""

# ╔═╡ 1c2ca675-0b92-40c6-ac30-56d039b42a0b
# md"""

# #### *To make it complete*:

# $$\large \begin{align}
# P(Y_2=\texttt h|Y_1=\texttt t) 
# &= \sum_c {P(C=c)P(Y_1=\texttt t|C=c)P(Y_2=\texttt h|C=c)}\\
# &= 0.5 \cdot 0.5 \cdot 0.5 + 0.5 \cdot 0.99 \cdot 0.01 \approx 0.13
# \end{align}$$


# $$\large\begin{align}
# P(Y_2=\texttt t|Y_1=\texttt t) 
# &= \sum_c {P(C=c)P(Y_1=\texttt t|C=c)P(Y_2=\texttt t|C=c)}\\
# &= 0.5 \cdot 0.5 \cdot 0.5 + 0.5 \cdot 0.99 \cdot 0.99 \approx 0.62
# \end{align}$$


# #### After normalisation,


# ```math
# \Large
# P(Y_2|Y_1=\texttt{t}) =\begin{cases} 0.174 & Y_2= \texttt h \\ 0.826 & Y_2=\texttt t\end{cases}

# ```

# """

# ╔═╡ 0c088fa5-d385-4d16-a7f8-30f378072107
md"""


## What `sum-out` is doing ?





$$\Large \begin{align}P(Y_2|Y_1=y_1) &\propto P(Y_1=y_1, Y_2) \\
&= \sum_{c=1,2} P(C=c, Y_1 =y_1, Y_2) \\
&= \sum_{c=1,2} \underbrace{P(C=c){P(y_1|C=c)}}_{\propto\; P(C=c|Y_1=y_1)}P(Y_2|C=c) 
\end{align}$$ 


* ##### Bayes' rule: ``P(C=c|Y_1=y_1) \propto P(C=c)\,{P(Y_1=y_1|C=c)}``

"""

# ╔═╡ 75b7abcd-6556-41d7-96f4-0432db64635c
md"""


## Why `sum-out` is better ?





$$\large \begin{align}P(Y_2|Y_1&=y_1) = \sum_{c=1,2} \underbrace{P(C=c){P(y_1|C=c)}}_{\propto\; P(C=c|Y_1=y_1)}P(Y_2|C=c) \\
&\propto \sum_{c=1,2} {P(C=c|Y_1=y_1)} P(Y_2|C=c) \\
&=\textcolor{red}{\underbrace{{P(C=1|Y_1=y_1)}}_{\color{red}\text{how likely coin 1 used?}}}\;  \textcolor{red}{\underbrace{\textcolor{red}{P(Y_2|C=1)}}_{\text{coin 1's prediction}}} \\
&\quad\quad\quad\quad\quad\quad\quad\quad+ \\
&\;\;\;\;\;\textcolor{blue}{\underbrace{{P(C=2|Y_1=y_1)}}_{\text{how likely coin 2 used?}} }\; \textcolor{blue}{\underbrace{ \textcolor{blue}{P(Y_2|C=2)}}_{\text{coin 2's prediction}}}
\end{align}$$ 


## Why `sum-out` is better ? (cont.)



$$\large \begin{align}P(Y_2|Y_1=y_1) &\propto \textcolor{red}{\underbrace{{P(C=1|Y_1=y_1)}}_{\color{red}\text{how likely coin 1 used?}}}\;  \textcolor{red}{\underbrace{\textcolor{red}{P(Y_2|C=1)}}_{\text{coin 1's prediction}}} \\
&\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;+ \\
&\;\;\;\;\;\textcolor{blue}{\underbrace{{P(C=2|Y_1=y_1)}}_{\text{how likely coin 2 used?}} }\; \textcolor{blue}{\underbrace{ \textcolor{blue}{P(Y_2|C=2)}}_{\text{coin 2's prediction}}}
\end{align}$$ 

#### Interpretation of ``P(Y_2|Y_1=y_1)``: a _weighted average_

* ##### both coin 1 and 2 are used to predict $Y_2$

$\Large\color{blue}{P(Y_2|C=1)};\; \color{red}{P(Y_2|C=2)}$


* ##### *the weights* are ``P(C|Y_1=y_1)``
$\Large\color{blue}P(C=1|Y_1=y_1);\; \color{red} P(C=2|Y_1=y_1)$


* ##### both coin choices are used to make a _prediction_ and weighted by their posteriors
"""

# ╔═╡ 66bc9601-d7a0-428a-aba6-7b41bfc8ff17
md"""


## How to do better ?


### The answer is simple: 



> ### _DO NOT_ try to be clever 
> ### ``\quad\quad`` stick to the two probability rules 
> ### ``\quad\quad``(a.k.a. Bayesian inference)!


"""

# ╔═╡ 56454443-1031-4be1-af7b-d4317d777ed9
md"""

## How is it relevant to ML?


"""

# ╔═╡ 1d244049-014d-4b74-bc92-799c598fbb7f
md"""

## Recap: Bayesian logistic regression as a BN

"""

# ╔═╡ c3ffe037-4be7-410a-a114-86f581a3a1da
TwoColumn(md"""


#### The CPF for ``\mathbf{w}`` : e.g. a uniform prior

```math
\large p(\mathbf{w}) \propto 1
```

* ##### for all $\mathbf{w} \in \mathbb{R}^m$



#### The CPF for ``y^{(i)}`` given its parents: ``\mathbf{x}^{(i)}, \mathbf{w}``:
* ##### Bernoulli with a bias

```math
\large p(y^{(i)}|\mathbf{x}^{(i)}, \mathbf{w}) = \begin{cases} \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 1\\ 1- \sigma(\mathbf{w}^\top \mathbf{x}^{(i)})& y^{(i)} = 0\end{cases} 
```



""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_first.png' width = '180' /></center>")

# ╔═╡ 9735a0cd-335a-47c9-8014-8ba68a37a0c7
md"""

## Recap: Bayesian logistic regression


#### ``\large \mathbf{w}`` determines 

```math
\Large P(y|\mathbf{x}, \mathbf{w}) 
```

"""

# ╔═╡ 3c56071f-ec06-43e4-8394-47f7f45eb206
md"Change ``w_1``: $(@bind ww1_ Slider(-1.5:0.01:1.8; default=0.5)) Change ``w_2``: $(@bind ww2_ Slider(-1.5:0.01:1.8; default=0.5)), change angle $(@bind theta Slider(-180:180; default=58))"

# ╔═╡ 4e5891f6-be11-47b1-8a6c-eb137eeff78c
md"Add decision boundary $(@bind add_db CheckBox())"

# ╔═╡ e813b082-3c70-46e6-85a2-4685e77e8f03
ww_ = [ww1_, ww2_]

# ╔═╡ ca0661f4-8e22-40ef-a626-03fe4ee15860
md"""


## Bayesian prediction


"""

# ╔═╡ 7ee2350e-276d-4d0e-acd2-a1cf3ef49c3e
TwoColumn(md"""


#### Predict $y_{test}$ given ``\mathbf{x}_{test}``?
* ##### add a hidden $y_{test}$ node
\

#### And compute preditive distribution 

$\Large p(y_{test}|\{y^{(i)}\}_{i=1}^n, \mathbf{x}_{test})$
""", html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/logistic_pred.png' width = '200' /></center>")

# ╔═╡ f5424411-95f2-4eb7-ab5d-2341403aa2ec
md"""

## Bayesian prediction (cont.)


#### Apply exact inference (sum-out nuisance ``\mathbf{w}``)


```math
\Large \begin{align}
p(y_{test} &|\mathbf{x}_{test}, \{y^{(i)}\}_{i=1}^n) \propto p(y_{test}, \mathbf{x}_{test}, \{y^{(i)}\}_{i=1}^n) \\
&\;\;\;\;\;\; \;\;\;\;\;\; \;\;\;\;\;\; \;\;\;\;\;\; \;\; \vdots \\
&= \sum_{\text{all possible }\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  \underbrace{p(\mathbf{w}|\{y^{(i)}\}_{i=1}^n)}_{\text{the weight}}
\end{align}
```


#### Again it is a weighted average of all possible predictions

* ##### ensemble method: democratic and balanced
* ##### we do not know the true ``\mathbf{w}``: we may as well just let every ``\mathbf{w} \in \mathbb{R}^m`` to predict then take average
  * and the average is weighted by ``p(\mathbf{w}|\mathbf{y})``
"""

# ╔═╡ 6b9e37bb-bfb5-4d84-a896-cc454efabef3
Foldable("Skipped details", md"""



```math
\begin{align}
p(y_{test} |\mathbf{x}_{test}, \mathbf{y}) &\propto p(y_{test}, \mathbf{x}_{test}, \mathbf{y}) \\
&= \sum_{\mathbf{w}}p(\mathbf{w}, y_{test}, \mathbf{x}_{test}, \mathbf{y}) \;\;\; \text{should've really been integration}\\
&= \sum_{\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w}) \underbrace{p(\mathbf{y}| \mathbf{w})p(\mathbf{w})}_{\propto p(\mathbf{w}|\mathbf{y})}\\
&\propto \sum_{\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  p(\mathbf{w}|\mathbf{y})
\end{align}
```


* ``\mathbf{y} = \{y^{(i)}\}_{i=1}^n``
""")

# ╔═╡ 3f322c41-5d87-4cbf-a3ec-6cb3cf25398a
md"""

## Demonstration


```math
\begin{align}
p(y_{test} &|\mathbf{x}_{test}, \{y^{(i)}\}_{i=1}^n) \propto \sum_{\text{all possible }\mathbf{w}} p(y_{test}|\mathbf{x}_{test}, \mathbf{w})  {p(\mathbf{w}|\{y^{(i)}\}_{i=1}^n)}
\end{align}
```


* ##### below, we sample $\mathbf{w} \sim p(\mathbf{w}|\{y^{(i)}\}_{i=1}^n)$ and show their predictions $p(y_{test}|\mathbf{x}_{test}, \mathbf{w})$
"""

# ╔═╡ 4833a1a0-f7dc-4a78-aae1-711ca64b04e1
md"""``\mathbf{w}\sim p(\mathbf{w}|\{y^{(i)}\})``$(@bind wi Slider(1000:50:2000)); show decision boundary: $(@bind show_w_db CheckBox());  """



# ╔═╡ de7f04b8-0985-4aca-874a-36c0be0ba94a
md"""
show all: $(@bind show_all CheckBox());show decision boundaries: $(@bind show_all_db CheckBox()); show average: $(@bind show_average CheckBox()); change viewing angle $(@bind theta_ Slider(-180:180; default=58))"""

# ╔═╡ 5a27a4b5-b62c-47c3-b64e-c2231e30c9a8
md"""
## The end product
"""

# ╔═╡ 02ffc227-37d2-4c02-bf73-9c3f8fa269a8
# let

# 	p_bayes_pred = Plots.plot(D1[:, 1], D1[:,2], seriestype=:scatter, markersize = 5, markercolor=2, label="class 1", legend=:topright, xlim=[-1, 11], ylim=[-1,11])
# 	Plots.plot!(p_bayes_pred, D2[:, 1], D2[:,2], seriestype=:scatter, markersize = 5,  markercolor=1, label="class 2", ratio=1)
# 	mean_pred = mean(chain_array, dims=1)[:]

# 	b, k =  -chain_array[1, 1:2] ./	chain_array[1, 3]
# 	plot!(-2:12, (x) -> k*x+b,  lw=0.1, lc=:gray, label=L"\mathbf{w}\sim p(\mathbf{w}|\mathbf{y})")
# 	for i in 1000:5:2000
# 		b, k =  -chain_array[i, 1:2] ./	chain_array[i, 3]
# 		plot!(-2:12, (x) -> k*x+b,  lw=0.2, ls=:dash, lc=:gray, label="", title="Bayesian ensemble predictions: decision boundary view")
# 	end
# 	# b, k =  - mean_pred[1:2] ./	mean_pred[3]
# 	# plot!(-2:12, (x) -> k*x+b,  lw=3, lc=3, label=L"\texttt{mean}(\sigma^{(r)})")
# 	p_bayes_pred

# end

# ╔═╡ 8acdeffb-b9e4-476a-ac05-07aa02d496b4
md"""

## Appendix
"""

# ╔═╡ 36ce371c-c728-4524-987d-547be3089da9
md"""
#### Data generation
"""

# ╔═╡ 531456a8-692a-4f0e-ad97-864f9298a7e2
D₂, targets_D₂, targets_D₂_=let
	Random.seed!(123)
	D_class_1 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .+2
	D_class_2 = rand(MvNormal(zeros(2), Matrix([1 -0.8; -0.8 1.0])), 30)' .-2
	data₂ = [D_class_1; D_class_2]
	D₂ = [ones(size(data₂)[1]) data₂]
	targets_D₂ = [ones(size(D_class_1)[1]); zeros(size(D_class_2)[1])]
	targets_D₂_ = [ones(size(D_class_1)[1]); -1 *ones(size(D_class_2)[1])]
	D₂, targets_D₂,targets_D₂_
end

# ╔═╡ 52844f1a-0515-439d-a722-a5e385b5d40c
let
	gr()
	plt = scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	f̂(x, y) = logistic(0 + ww_[1] * x + ww_[2] * y)
	plot!(-8:.5:8, -8:0.5:8, (x,y) -> f̂(x, y), alpha =0.4, st=:surface, colorbar=false,c=:jet, title="", camera=(theta,20), xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y")

	for (i, xi) in enumerate(eachrow(D₂[:, 2:3]))
		if targets_D₂[i] == 1
			plot!([xi[1], xi[1]], [xi[2], xi[2]], [1, f̂(xi...)], lw=0.2, c =(f̂(xi...) >0.5) ? 2 : 1, label="")

		else
			plot!([xi[1], xi[1]], [xi[2], xi[2]], [0, f̂(xi...)], lw =0.2, c=(f̂(xi...) >0.5) ? 2 : 1, label="")
		end
	end

	# w₀ = 0
	# wv_ = - ww_
	# w₁, w₂ = wv_[1], wv_[2]
	# x0s = -3:0.5:3
	# if w₂ ==0
	# 	x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
	# 	y0s = range(-5, 5, 20)
	# else
	# 	y0s = (- w₁ * x0s .- w₀) ./ w₂
	# end
	# plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=2, label="")
	if add_db
		plot!(-8:1:8, -8:1:8, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.3)
	end
	plt
end

# ╔═╡ 8e1c0744-e813-4f53-af1a-27d27ca58769
begin
	D1 = [
	    7 4;
	    5 6;
	    8 6;
	    9.5 5;
	    9 7
	]

	# D1 = randn(5, 2) .+ 2
	
	D2 = [
	    2 3;
	    3 2;
	    3 6;
	    5.5 4.5;
	    5 3;
	]

	# D2 = randn(5,2) .- 2

	D = [D1; D2]
	D = [ones(10) D]
	targets = [ones(5); zeros(5)]
	AB = [1.5 8; 10 1.9]
end;

# ╔═╡ 78060a5a-8201-4443-9a45-5b8390c9e702
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ f448ee19-f495-45c5-957d-45e15ac27f5f
# as: arrow head size 0-1 (fraction of arrow length)
# la: arrow alpha transparency 0-1
function arrow3d!(x, y, z,  u, v, w; as=0.1, lc=:black, la=1, lw=0.4, scale=:identity)
    (as < 0) && (nv0 = -maximum(norm.(eachrow([u v w]))))
    for (x,y,z, u,v,w) in zip(x,y,z, u,v,w)
        nv = sqrt(u^2 + v^2 + w^2)
        v1, v2 = -[u,v,w]/nv, nullspace(adjoint([u,v,w]))[:,1]
        v4 = (3*v1 + v2)/3.1623  # sqrt(10) to get unit vector
        v5 = v4 - 2*(v4'*v2)*v2
        (as < 0) && (nv = nv0) 
        v4, v5 = -as*nv*v4, -as*nv*v5
        plot!([x,x+u], [y,y+v], [z,z+w], lc=lc, la=la, lw=lw, scale=scale, label=false)
        plot!([x+u,x+u-v5[1]], [y+v,y+v-v5[2]], [z+w,z+w-v5[3]], lc=lc, la=la, lw=lw, label=false)
        plot!([x+u,x+u-v4[1]], [y+v,y+v-v4[2]], [z+w,z+w-v4[3]], lc=lc, la=la, lw=lw, label=false)
    end
end

# ╔═╡ f1c3de03-6f24-4dd7-be37-3550118e3e7e
begin
	# monte carlo prediction
	# ws: MCMC samples, assumed a mc × 3 vector
	# demonstration for 2-d case
	prediction(ws, x1, x2) = mean(logistic.(ws * [1.0, x1, x2]))
end

# ╔═╡ 430c7352-8857-41bb-b919-7e330f9b84ff
begin
	function logistic_loss(w, X, y)
		σ = logistic.(X * w)
		# deal with boundary cases such as σ = 0 or 1, log(0) gracefully
		# sum(y .* log.(σ) + (1 .- y).* log.(1 .- σ))
		# rather you should use xlogy and xlog1py
		-sum(xlogy.(y, σ) + xlog1py.(1 .-y, -σ))
	end
end

# ╔═╡ 5078f341-3190-40c0-8952-2fbbd8167c5b
begin
	function MCMCLogisticR(X, y, dim; mc= 1000, burnin=0, thinning=10, m₀= zeros(dim), λ = 1/1e2, qV= nothing)
		if isnothing(qV)
			qV = Symmetric(inv(6/(π^2) * X' * X))
		end
		postLRFun(w) = -(logistic_loss(w, X, y) + 0.5 * λ * (w' * w))
		mcLR = MHRWMvN((w) -> postLRFun(w), dim; logP = true, Σ = qV, x0=m₀, mc=mc, burnin=burnin, thinning= thinning)
		return mcLR
		# return wt, Ht
	end


	# Metropolis Hastings with simple Gaussian proposal
	function MHRWMvN(pf, dim; logP = true, Σ = 10. *Matrix(I, dim, dim), x0=zeros(dim), mc=5000, burnin =0, thinning = 1)
		samples = zeros(dim, mc)
		C = cholesky(Σ)
		L = C.L
		pfx0 = pf(x0) 
		j = 1
		for i in 1:((mc+burnin)*thinning)
			xstar = x0 + L * randn(dim)
			pfxstar = pf(xstar)
			if logP
				Acp = pfxstar - pfx0 
				Acp = exp(Acp)
			else
				Acp = pfxstar / pfx0 
			end
			if rand() < Acp
				x0 = xstar
				pfx0 = pfxstar
			end
			if i > burnin && mod(i, thinning) ==0
				samples[:,j] = x0
				j += 1
			end
		end
		return samples
	end
	
end

# ╔═╡ 4c72a682-1b1d-4c64-afc3-a61fee9b37d1
mcmcLR = MCMCLogisticR(D, targets, 3; mc=2000);

# ╔═╡ f0cda217-6a2c-4119-9174-c53eb28b78d4
begin
	gr()
	ppf(x, y) = prediction(mcmcLR[:,1:2000]', x, y)
	contour(-0:0.1:10, 0:0.1:10, ppf, xlabel=L"x_1", ylabel=L"x_2", fill=true,  connectgaps=true, line_smoothing=0.85, title="Bayesian prediction", c=:jet, alpha=0.4, ratio=1, xlim =(0,10), frame=false)
end

# ╔═╡ da9325bd-28e9-4317-aba4-0e2a338b6cff
chain_array = mcmcLR';

# ╔═╡ 7e4c6c05-16a3-4914-ab94-0730c965e424
let
	gr()
	plt = scatter(D[targets .== 1, 2], D[targets .== 1, 3], ones(sum(targets .== 1)),  label="y=1", c=2)
	scatter!(D[targets .== 0, 2], D[targets .== 0, 3], 0 *ones(sum(targets .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	f̂(x, y, ww = ww_, b =0) = logistic(b + ww[1] * x + ww[2] * y)


	b, w = chain_array[wi, 1], chain_array[wi, 2:3]
	plot!(0:1:10, 0:1:10, (x,y) -> f̂(x, y, w, b), alpha =0.3, st=:surface, c=:jet, colorbar=false, camera=(theta_,20), xlabel=L"x_1", ylabel=L"x_2", xlim =(0,10), ylim = (0,10))

	w₀, w₁, w₂ = b, w[1], w[2]
	x0s = 0:1:10
	if w₂ ==0
		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
		y0s = range(-5, 5, 20)
	else
		y0s = (- w₁ * x0s .- w₀) ./ w₂
	end
	if show_w_db
		plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=2, label="")
	
	end

	
	
	for (ci, i) in enumerate(500:100:2000)
		b, w = chain_array[i, 1], chain_array[i, 2:3]
		if show_all
			plot!(0:0.5:10, 0:.5:10, (x,y) -> f̂(x, y, w, b), alpha =0.15, st=:surface, c=:gray, colorbar=false)
		end
		if show_all_db
			w₀, w₁, w₂ = b, w[1], w[2]
			# x0s = 0:0.5:10
			if w₂ ==0
				x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
				y0s_ = range(-5, 5, 20)
			else
				y0s_ = (- w₁ * x0s .- w₀) ./ w₂
			end
		
			plot!(x0s, y0s_, .5 * ones(length(x0s)), lc=ci, lw=1, label="")
		end
	end



	if show_average

		ppf(x, y) = prediction(mcmcLR[:,1000:2000]', x, y)
		# for (ci, i) in enumerate(1000:200:2000)
			# b, w = chain_array[i, 1], chain_array[i, 2:3]
		plot!(0:.5:10, 0:0.5:10, (x,y) -> ppf(x, y), alpha =0.25, st=:surface, c=:jet, colorbar=false)
		# end

	end
		# plot!(-2:12, (x) -> k*x+b,  lw=0.2, ls=:dash, lc=:gray, label="", title="Bayesian ensemble predictions")
	# end

	# for (i, xi) in enumerate(eachrow(D₂[:, 2:3]))
	# 	if targets_D₂[i] == 1
	# 		plot!([xi[1], xi[1]], [xi[2], xi[2]], [1, f̂(xi...)], lw=0.2, c =(f̂(xi...) >0.5) ? 2 : 1, label="")

	# 	else
	# 		plot!([xi[1], xi[1]], [xi[2], xi[2]], [0, f̂(xi...)], lw =0.2, c=(f̂(xi...) >0.5) ? 2 : 1, label="")
	# 	end
	# end

	# if add_db
	# 	plot!(-8:1:8, -8:1:8, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.3)
	# end
	plt
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TikzGraphs = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"

[compat]
Distributions = "~0.25.117"
Graphs = "~1.12.0"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.29"
Plots = "~1.34.3"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.43"
TikzGraphs = "~1.4.0"
TikzPictures = "~3.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "81d620e062595efef440de65009f07a566806ade"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.ArnoldiMethod]]
deps = ["LinearAlgebra", "Random", "StaticArrays"]
git-tree-sha1 = "d57bd3762d308bded22c3b82d033bff85f6195c6"
uuid = "ec485272-7323-5ecc-a04f-4719b315124d"
version = "0.4.0"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "403f2d8e209681fcbd9468a8514efff3ea08452e"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.29.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "88d48e133e6d3dd68183309877eac74393daa7eb"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.17.20"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "03aa5d44647eaec98e1920635cdfed5d5560a8b9"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.117"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d55dffd9ae73ff72f1c0482454dcf2ec6c6c4a63"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.5+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Pkg", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "74faea50c1d007c85837327f6775bea60b5492dd"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.2+2"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Printf", "Random", "RelocatableFolders", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "0ac6f27e784059c68b987f42b909ade0bcfabe69"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.68.0"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Pkg", "Qt5Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "bc9f7725571ddb4ab2c4bc74fa397c1c5ad08943"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.69.1+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "Compat", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "1dc470db8b1131cfc7fb4c115de89fe391b9e780"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.12.0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_ICU_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "HarfBuzz_jll", "ICU_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "6ccbc4fdf65c8197738c2d68cc55b74b19c97ac2"
uuid = "655565e8-fb53-5cb3-b0cd-aec1ca0647ea"
version = "2.8.1+0"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg"]
git-tree-sha1 = "129acf094d168394e80ee1dc4bc06ec835e510a3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "2.8.1+1"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "2bd56245074fab4015b9174f24ceba8293209053"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.27"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.ICU_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "20b6765a3016e1fca0c9c93c80d50061b94218b7"
uuid = "a51ab1cf-af8e-5615-a023-bc2c838bba6b"
version = "69.1.0+0"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eac1206917768cb54957c65a615460d87b455fc1"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.1+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "4bf4b400a8234cff0f177da4a160a90296159ce9"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.41"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "50901ebc375ed41dbf8058da26f9de442febbbec"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.3.1"

[[deps.Latexify]]
deps = ["Formatting", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Printf", "Requires"]
git-tree-sha1 = "8c57307b5d9bb3be1ff2da469063628631d4d51e"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.15.21"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    DiffEqBiologicalExt = "DiffEqBiological"
    ParameterizedFunctionsExt = "DiffEqBase"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    DiffEqBase = "2b5f629d-d688-5b77-993f-72d75c75574e"
    DiffEqBiological = "eb300fae-53e8-50a0-950c-e21f52c2b7e0"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "df37206100d39f79b3376afb6b9cee4970041c61"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.1+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89211ea35d9df5831fca5d33552c02bd33878419"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.3+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e888ad02ce716b319e6bdb985d2ef300e7089889"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.3+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pkg"]
git-tree-sha1 = "110897e7db2d6836be22c18bffd9422218ee6284"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.12.0+0"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

[[deps.LoweredCodeUtils]]
deps = ["JuliaInterpreter"]
git-tree-sha1 = "688d6d9e098109051ae33d126fcfc88c4ce4a021"
uuid = "6f1432cf-f94c-5a45-995e-cdbf5db27b0b"
version = "3.1.0"

[[deps.MIMEs]]
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.MacroTools]]
git-tree-sha1 = "72aebe0b5051e5143a079a4685a46da330a40472"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.15"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2023.12.12"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "cc0a5deefdb12ab3a096f00a6d42133af4560d71"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ad31332567b189f508a3ea8957a2640b1147ab00"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "1.1.23+1"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "966b85253e959ea89c53a9abebbf2e964fbf593b"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.32"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9dd97171646850ee607593965ce1f55063d8d3f9"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.0+0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SnoopPrecompile", "SparseArrays", "Statistics", "StatsBase", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "284a353a34a352a95fca1d61ea28a0d48feaf273"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.34.4"

[[deps.PlutoHooks]]
deps = ["InteractiveUtils", "Markdown", "UUIDs"]
git-tree-sha1 = "072cdf20c9b0507fdd977d7d246d90030609674b"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0774"
version = "0.0.5"

[[deps.PlutoLinks]]
deps = ["FileWatching", "InteractiveUtils", "Markdown", "PlutoHooks", "Revise", "UUIDs"]
git-tree-sha1 = "8f5fa7056e6dcfb23ac5211de38e6c03f6367794"
uuid = "0ff47ea0-7a50-410d-8455-4348d5de0420"
version = "0.1.6"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "LaTeXStrings", "Latexify", "Markdown", "PlutoLinks", "PlutoUI", "Random"]
git-tree-sha1 = "5d9ab1a4faf25a62bb9d07ef0003396ac258ef1c"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.2.15"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.Poppler_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "02148a0cb2532f22c0589ceb75c110e168fb3d1f"
uuid = "9c32591e-4766-534b-9725-b71a8799265b"
version = "21.9.0+0"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Revise]]
deps = ["CodeTracking", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "9bb80533cb9769933954ea4ffbecb3025a783198"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.7.2"
weakdeps = ["Distributed"]

    [deps.Revise.extensions]
    DistributedExt = "Distributed"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "e3be13f448a43610f978d29b7adf78c76022467a"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.12"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TikzGraphs]]
deps = ["Graphs", "LaTeXStrings", "TikzPictures"]
git-tree-sha1 = "e8f41ed9a2cabf6699d9906c195bab1f773d4ca7"
uuid = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
version = "1.4.0"

[[deps.TikzPictures]]
deps = ["LaTeXStrings", "Poppler_jll", "Requires", "tectonic_jll"]
git-tree-sha1 = "79e2d29b216ef24a0f4f905532b900dcf529aa06"
uuid = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
version = "3.5.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "b8b243e47228b4a3877f1dd6aee0c5d56db7fcf4"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.6+1"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+3"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e9216fdcd8514b7072b43653874fd688e4c6c003"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.12+0"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "89799ae67c17caa5b3b5a19b8469eeee474377db"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.5+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+3"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c57201109a9e4c0585b208bb408bc41d205ac4e9"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.2+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+3"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6dba04dbfb72ae3ebe5418ba33d087ba8aa8cb00"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.1+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "522c1df09d05a71785765d19c9524661234738e9"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.11.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Pkg", "Zlib_jll"]
git-tree-sha1 = "5982a94fcba20f02f42ace44b9894ee2b140fe47"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.1+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "055a96774f383318750a1a5e10fd4151f04c29c5"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.46+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.tectonic_jll]]
deps = ["Artifacts", "Fontconfig_jll", "FreeType2_jll", "Graphite2_jll", "HarfBuzz_ICU_jll", "HarfBuzz_jll", "ICU_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "54867b00af20c70b52a1f9c00043864d8b926a21"
uuid = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"
version = "0.13.1+0"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╟─c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
# ╟─cf07faa6-e571-4c79-9dd2-f03041b99706
# ╟─3f49be0c-84d6-4bb1-adff-ab7f7ce15d05
# ╟─cb9278ae-fd3f-4a2d-b188-ab463006db38
# ╟─35494406-218b-41cd-b3af-6ba07d45d935
# ╟─9179c13b-1cc4-4ec6-91f2-aaed6a312ee9
# ╟─e67e822d-25c6-44a7-8ae2-13f42534d70b
# ╟─eac84f66-a017-436e-b3a0-30a32677179e
# ╟─ed09a8e6-6b8f-415d-84f2-bcaed9d32558
# ╟─4f58814a-e990-445c-b959-f92d9aac1fe9
# ╟─1769f7b4-4212-487a-bad3-bfaa3bf7026c
# ╟─89be5e5a-76e6-446a-aa7a-00d0cccd0c33
# ╟─5d45e5db-6b95-48c0-8471-c062c38aab2b
# ╟─4edc91cb-89d6-45df-8055-c4752189bf7a
# ╟─32770f48-6f30-435a-86b2-b1ad272b29ff
# ╟─7518fc3a-685c-4089-9be8-1edb332a68d2
# ╟─8a0f33b2-28d1-475e-a30f-289fec3b59bd
# ╟─96c86194-fe8a-444d-ae43-fc98ad6e5181
# ╟─22134218-1606-45b0-89e2-f199bb17b15b
# ╟─8c2f8f52-c440-4ad3-a08e-290156552c08
# ╟─bbbd12d6-ec08-482c-a82b-e20161ebd507
# ╟─2ac736f1-b27c-46a4-8ae4-d929a2efa5d8
# ╟─c7a8c934-e45d-4f34-9429-b7947cea95f8
# ╟─0ab4585d-9d91-4457-a112-6df073c1f4c0
# ╟─e2a88805-3f01-4ab3-8b9e-a3e4012b49b7
# ╟─3131731b-f6de-43d5-bf78-8b1f321a04c5
# ╟─84481277-4b3f-4c9c-9181-a32e76fc8546
# ╟─16c3e05c-35f1-4857-bbb4-e0feed7507fd
# ╟─514f8322-f2f4-4384-8661-3c7683d21a78
# ╟─469b80f9-26db-4d16-9980-7c7e100ba486
# ╟─86c2dd5d-d7c6-4f4b-a793-06aacc15375c
# ╟─28611165-f29b-4163-bf02-796598804129
# ╟─42e93176-c83d-49af-91b8-f01cda668edd
# ╟─693e0834-a970-44e4-a273-b71233984d70
# ╟─bc2b6d66-ceb0-4fcb-8af7-53c3dde09f20
# ╟─27d8fc53-e121-4e3b-9861-77fcbaf00cd3
# ╟─74125061-4bd4-41d5-aeeb-436074db9ded
# ╟─9c58d982-b26a-4d98-9d04-c07ee3816e4a
# ╟─9f49f45d-e350-480e-a2c0-2fb48ada6d88
# ╟─2a7462da-7a1a-4ab7-8544-5e2be106fd43
# ╟─dfd22bfb-ff0a-4847-8cb1-24ef1d24f3c9
# ╟─ce48b420-bb2b-4a85-a875-2084897b0134
# ╟─ac943f36-669e-4799-8b31-6c8d01c328ba
# ╟─8e330eb7-7790-438e-bce6-9c02b45a8dc9
# ╟─ee50d06e-eb24-49f1-9f56-846b097a3238
# ╟─75cb72a8-7620-4b63-9a48-85414141354c
# ╟─0406ae12-8be2-478f-993f-10837abfcadf
# ╟─67a909b7-ca69-4d0d-9a74-1d0622fb8397
# ╠═5d8f7bfc-7192-4ce4-a523-15abc80047dd
# ╠═2a2437d6-b774-42f8-887e-6b5d4940b367
# ╠═e13c61c8-f0a8-49be-a523-9aca895640f2
# ╠═b26dd8ed-bae2-4454-af53-31711fad8605
# ╟─47e5143b-b7d5-4ba9-a0fa-b1abd74b1bde
# ╟─4a4a91c0-5c25-4e54-99e1-c731c53f4972
# ╟─98ebe05d-dd1f-4b6b-afab-e7f42606d5bb
# ╠═75c4a259-7d39-49bf-bda8-1e5a89770598
# ╟─a4d35349-ac25-4e47-adfc-c31e6798e265
# ╟─a9d05d1b-f789-4a7f-80bf-767cd0981331
# ╟─2cd4b610-2fbc-43ef-8c64-1480e60356ba
# ╟─430cc0b0-4748-4900-be65-22c1f8b0ec02
# ╟─d0ea7686-9d39-40d3-a695-8c71783a4a1f
# ╟─748ade25-bb47-4a67-a624-663fb7a5933f
# ╟─344a3078-fad8-4985-8270-c52b7be034f2
# ╟─56013389-2242-4f23-b272-558fcdecd3b1
# ╟─409d6b4d-f0c3-44b0-bfab-7331f434de9b
# ╟─8bd9e5a8-4534-4c2b-b6e6-5736bd9ec745
# ╟─b44bedea-a0d8-497c-9285-ab5f7aed6d60
# ╟─36c02a04-f925-4bfe-a530-b63d89b5e620
# ╟─f8a835e6-55f7-42f1-84c2-ee977d70dc38
# ╟─56acc259-7bc1-4f40-8745-b41950fba0ac
# ╟─2de5daf9-b422-43fb-a22a-7c3c68ae0bf1
# ╟─2502e421-e8a4-4712-9cb3-3ada7646edfb
# ╟─9bbf617a-d355-48cd-a150-c276d9e61a85
# ╟─42141d4d-3a11-4240-acb0-91bf412962f8
# ╟─1c2ca675-0b92-40c6-ac30-56d039b42a0b
# ╟─0c088fa5-d385-4d16-a7f8-30f378072107
# ╟─75b7abcd-6556-41d7-96f4-0432db64635c
# ╟─66bc9601-d7a0-428a-aba6-7b41bfc8ff17
# ╟─56454443-1031-4be1-af7b-d4317d777ed9
# ╟─1d244049-014d-4b74-bc92-799c598fbb7f
# ╟─c3ffe037-4be7-410a-a114-86f581a3a1da
# ╟─9735a0cd-335a-47c9-8014-8ba68a37a0c7
# ╟─3c56071f-ec06-43e4-8394-47f7f45eb206
# ╟─4e5891f6-be11-47b1-8a6c-eb137eeff78c
# ╟─e813b082-3c70-46e6-85a2-4685e77e8f03
# ╟─52844f1a-0515-439d-a722-a5e385b5d40c
# ╟─ca0661f4-8e22-40ef-a626-03fe4ee15860
# ╟─7ee2350e-276d-4d0e-acd2-a1cf3ef49c3e
# ╟─f5424411-95f2-4eb7-ab5d-2341403aa2ec
# ╟─6b9e37bb-bfb5-4d84-a896-cc454efabef3
# ╟─3f322c41-5d87-4cbf-a3ec-6cb3cf25398a
# ╟─4833a1a0-f7dc-4a78-aae1-711ca64b04e1
# ╟─de7f04b8-0985-4aca-874a-36c0be0ba94a
# ╟─7e4c6c05-16a3-4914-ab94-0730c965e424
# ╟─5a27a4b5-b62c-47c3-b64e-c2231e30c9a8
# ╟─f0cda217-6a2c-4119-9174-c53eb28b78d4
# ╟─02ffc227-37d2-4c02-bf73-9c3f8fa269a8
# ╟─8acdeffb-b9e4-476a-ac05-07aa02d496b4
# ╟─36ce371c-c728-4524-987d-547be3089da9
# ╟─531456a8-692a-4f0e-ad97-864f9298a7e2
# ╠═8e1c0744-e813-4f53-af1a-27d27ca58769
# ╠═78060a5a-8201-4443-9a45-5b8390c9e702
# ╠═f448ee19-f495-45c5-957d-45e15ac27f5f
# ╠═5078f341-3190-40c0-8952-2fbbd8167c5b
# ╠═4c72a682-1b1d-4c64-afc3-a61fee9b37d1
# ╠═da9325bd-28e9-4317-aba4-0e2a338b6cff
# ╠═f1c3de03-6f24-4dd7-be37-3550118e3e7e
# ╠═430c7352-8857-41bb-b919-7e330f9b84ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
