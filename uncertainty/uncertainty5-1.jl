### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    return quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 3a68dad0-0016-4316-8846-ccb5425658ad
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

	# using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ 9428adb1-48f6-49d1-9961-2fd7e95127ce
begin
	using TikzGraphs
	using Graphs
	using TikzPictures # this is required for saving
end;

# ╔═╡ 8e953cf6-e9bd-4ab7-b332-1706824c6e91
TableOfContents()

# ╔═╡ 34c34925-daed-42fe-bdc2-1cf040fd4711
ChooseDisplayMode()

# ╔═╡ c28f13a8-c2c5-4f50-b092-33ec1f7d7516
md"""


# CS3105 Artificial Intelligence


### Uncertainty 5
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ f265348c-e69f-499d-8ebb-634416d6b14e
md"""

# Probabilistic inference algorithm
\

### Exact inference algorithm
"""

# ╔═╡ 96c1556d-b398-486e-a920-791426c8a09a
md"""

## Recep: Bayes' net

"""

# ╔═╡ f1fe3dd3-98de-4288-b1d8-affd3d190584
TwoColumnWideRight(md"""
#### Bayes' net 

$\Large \text{bn} = \{G, \{P\}\}$

* ##### ``G``: _a DAG_
 


* ##### CPTs: for each node ``\large P(X_i|\text{parent}(X_i))`` 


""", html""" <center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglarCPTs.png" width = "700"/></center>
""")

# ╔═╡ 0033eac7-c9dd-43af-b83d-25c99dca82e4
md"""

## Recap: factoring property

"""

# ╔═╡ 94c0287f-29f9-437f-b252-c8c903531962
md"""

!!! important "Factoring property"
	$\Large P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$
	
"""

# ╔═╡ 2a50a817-9594-4bd3-a45f-5d9a69a2ec6b
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

# ╔═╡ 5768a0f8-3730-4d4c-a63f-00da4bc96567
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

# ╔═╡ c56947cb-0b9f-4b64-94e0-cf2e9e9b74dc
md"""

## Some inference examples



"""

# ╔═╡ 543a9a65-29ea-486e-b51f-9c5a7c4a0656
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

# ╔═╡ 8af7d7a2-0d16-43ef-9bb4-5ab12b0cfdef
# md"""


# ## Query types


# * **bottom-up**: given evidence (observations, data) infer the cause; opposite the direction of edges, 
#   * *e.g.* $P(B|J,M)$
# * **top-down**: given cause infer downstream r.v.s (follow the direction of edge), 
#   * prediction of future data
#   * *e.g.*  $P(J|B, E)$

# * **mixture of both**: $P(J|M)$

# """

# ╔═╡ ee52fb06-eeb7-481b-b330-4099a4205d0e
# html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "300"/></center>"""

# ╔═╡ dbe7ace2-f9be-44ad-aff6-3c10fc9861d9
md"""

## The inference formula


!!! information ""
	#### Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\Large \begin{align}
	P(Q|\mathcal{E}=\mathbf{e}) \\

	\end{align}
	```
	* #### ``Q``: query random variable 
	* #### ``\mathcal{E}``: conditioned evidence 

"""

# ╔═╡ 1eb5e94d-414a-4a90-bdfa-cf14e02150dd
md"""

## The inference formula


!!! information ""
	#### Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\Large \begin{align}
	P(Q|\mathcal{E}=\mathbf{e}) &\propto P(Q, \mathcal{E}=\mathbf{e}) \\
	&= \sum_{n_1}\ldots \sum_{n_k} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)
	\end{align}


	```
	* #### ``Q``: query random variable 
	* #### ``\mathcal{E}``: conditioned evidence 
	* #### ``\{n_1, n_2, \ldots, n_k\}``: nuisance (or hidden) random variables

"""

# ╔═╡ f70b2cbb-5575-49f6-8ecf-aff40d5479c2
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
	* #### ``\{n_1, n_2, \ldots, n_k\}``: nuisance (or hidden) random variables

"""

# ╔═╡ ddfb8066-258d-43e7-b024-600d3df08c35
md"""

## An example 

#### Let's consider 


$\Large P(J|+b, -e)$


!!! question ""
	#### *By conditional probability definition*
	```math
	\large P(J|+b, -e) = \frac{P(J, +b, -e)}{P(+b, -e)} = \frac{P(J, +b, -e)}{P(+j, +b, -e) + P(-j, +b, -e)}
	```

"""

# ╔═╡ f9053f31-663c-4b12-a4b2-768f64f271e1
md"""

## An example 

#### Let's consider 


$\Large P(J|+b, -e)$


!!! question ""
	#### *By conditional probability definition*
	```math
	\large P(J|+b, -e) = \frac{P(J, +b, -e)}{P(+b, -e)} = \frac{P(J, +b, -e)}{P(+j, +b, -e) + P(-j, +b, -e)}
	```

#### Note that the denominator (due to sum rule) is a **normalising constant**

```math
\large P(+b, -e) = P(+j, +b, -e) + P(-j, +b, -e)\tag{sum rule}
```

*  ##### and it can be computed from the numerator ``P(J, +b, -e)`` 


##

!!! question ""
	#### Therefore, as a short-hand notation, queries are often written as  

	$$\Large P(J|+b, -e) \propto P(J, +b, -e)$$ 

	#### or equivalently
	
	$$\Large P(J|+b, -e) =\alpha P(J, +b, -e)$$ 
	* ##### where ``\alpha = \frac{1}{P(+b, -e)}`` is the normalising constant's inverse


"""

# ╔═╡ c321eeee-2a4b-45a3-a40e-3a32aaa35cd8
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

# ╔═╡ e9f05552-2173-4d73-897c-71ee41b746d2
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

# ╔═╡ b68f0f17-9d28-4b90-a519-c76853098976
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



"""

# ╔═╡ 83f2d335-2df3-411e-a85a-48a37755120c
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

"""

# ╔═╡ d292ee9c-49d9-443a-ae64-42e485ed7825
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

# ╔═╡ 2b828f8f-1e1a-4ac0-b4d6-c8e2ef64adba
# md"""

# !!! question "Question"
# 	##### How many _multiplications_ involved?
# """

# ╔═╡ 366d22ca-e42d-48fb-8292-b7a5f8335ffe
# Foldable("", md"
# Roughly in the order

# ```math
# \Large 2^{N_k} \times \text{\# of factors} 
# ```

# * ##### ``N_k``: the number of nuisance variables
# * ##### ``\#`` of factors: or number of nodes in a BN
# * ##### ``O(2^{N_k}\cdot N)`` complexity
# ")

# ╔═╡ cf82fecc-ee7c-41a5-abb5-7ecb161333a6
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

# ╔═╡ f09f12ad-a97a-4f80-88e7-1173a33b7046
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

# ╔═╡ 75b3cecc-8729-4b67-b4b8-9b251e0cf6dd
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

# ╔═╡ a749ec19-ed6c-42ea-a6cd-d576f024e0cb
aside(tip(md"""

``\mathcal{N, Q, E}``: this font denotes sets of random variables

"""))

# ╔═╡ 9ef1dd1a-7d31-43c2-8d2a-57e4325bd92f
md"""

## A small improvement 


#### We can improve the algorithm a little bit

* ##### by removing some floating number multiplications


* ##### based on distribution law


$\large ax + ay = a(x+y)$
"""

# ╔═╡ e23b5d46-33e5-4a08-81aa-0cde9b551263
md"""
## Digress: summation notation

#### Recall the summation  

$\Large \sum_{i=1}^n ax_i =ax_1 + a x_2 + \ldots + ax_n= a\left (\sum_{i=1}^n x_i\right )$

* ##### when ``a`` is a common constant factor, 
  * it can be taken out or equivalently pushing summation $\,\Sigma_{\cdot}\,$ inwards
\

* ##### it saves some floating multiplication operations 
  * remember floating number ``\times`` is more expensive than ``+/-``



##

#### Example: ``P(J|+b, -e)``




"""

# ╔═╡ 74837ad6-a744-4758-9182-e1d4b883ed70
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

# ╔═╡ bab2725c-8ed1-44f2-b07c-9318d2d91385
md"""

##
#### That is 
$$\large \begin{align}

P(J|+b,-e)&= \alpha \sum_{a'}\sum_{m'} \underbrace{\boxed{P(+b)P(-e)}}_{\text{constant!}}P(A=a'|b,e)P(J|a')P(m'|a') \\
  &= \alpha P(+b)P(-e)\sum_{a'}\sum_{m'}{P(a'|b,e)P(J|a')}P(m'|a')
\end{align}$$



* ##### ``P(+b)P(-e)`` is a common factor




"""

# ╔═╡ c627354d-b66b-4525-8651-a831be1d237f
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

# ╔═╡ e287205c-565d-4b05-8ad4-4f3990081bbf
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

# ╔═╡ e84fbb5e-b289-4ae5-b8bc-554789a37573
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

# ╔═╡ 8fcce286-6a13-4238-925e-c7afe764c59d
md"""

## Another example



$$\large \begin{align}P&(B|j,m) = \alpha P(B,j,m)\;\;\; \text{conditional probability rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B,e',a',j,m)\;\;\; \text{summation rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B)P(e')P(a'|B,e')P(j|a')P(m|a') \;\;\; \text{factoring CPTs} \\
&= \alpha P(B)\sum_{e'}P(e')\sum_{a'} P(a'|B,e')P(j|a')P(m|a')\;\;\; \text{simplify} \end{align}$$


"""

# ╔═╡ dfdd07d7-121f-4dd4-b26f-273c6e4a688d
md"""

## How to implement the inference algorithm



#### So far, we have computed everything by hand
\

#### _How to implement it_?

"""

# ╔═╡ 2b2d21b5-6028-4c36-905a-f7cea4ae5ef1
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

# ╔═╡ 7eaaa6ef-f19d-405c-8f00-172d48972ae9
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

# ╔═╡ 5dcfcea3-ee02-42ff-bff4-a0182feb18e7
# md"""
# ## A naive inference algorithm

# For example, query 

# $P(J|B=\texttt{t},E=\texttt{t})$

# **Step 0**: populating the full joint table:


# """

# ╔═╡ b104c8b1-c4f9-4ac8-b25c-1aa7d4664d48
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

# ╔═╡ f08a2a91-6ff8-4d17-93c0-620d779418c4
# md"""

# **Step 1**: enumerate query variable ``J`` and apply sum rule

# * for ``J=\texttt{t}``, `filter` the table with the condition ``\{J=\texttt{t} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
#   * sum and store it in ``\texttt{vec}[J=\texttt{t}]``
# * for ``J=\texttt{f}``, `filter` the table with the condition ``\{J=\texttt{f} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
#   * sum and store it in ``\texttt{vec}[J=\texttt{f}]``

# **Step 2** normalise the ``\texttt{vec}`` and return it
# """

# ╔═╡ 97a9c33b-94bf-44ea-b510-057a5e9b9be4
# (tip(md"""

# What **step 1** essentially does (sum out the nuisances):

# $$\begin{align}&\texttt{vec}[J=t] =\sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=t, M=m')\end{align}$$


# $$\begin{align}&\texttt{vec}[J=f] = \sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=f, M=m')\end{align}$$

# """))

# ╔═╡ 6446ffaf-ce4f-4915-8121-ad5a7855b7f2
# md"""

# ## Naive algorithm does not scale
# """

# ╔═╡ 029e24d2-8524-4d3e-b6af-0dc76411965c
# md"""

# ##### *Step 0* is not feasible !
# \


# ###### The joint full table is **HUGE** 

# * ``2^N`` rows, ``N`` is the number of random variables in the BN
# * space complexity is ``O(2^{N})``


# The rest two steps, however, are plainly _simple and efficient_
# * just basic database `filter` operation and summation
# """

# ╔═╡ c3110f09-d152-4bf6-a08f-d6dd79538805
md"""

## Digress: enumerate all - Backtracking



#### Recursive backtracking! 
* ##### or recursive DFS tree search



"""

# ╔═╡ 0d163494-7c51-44bc-865b-436b0f3c6259
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/back_track.png" width = "650"/></center>"""

# ╔═╡ 14505638-8a59-4a8d-9799-ef86ea306e8d
md"""

#### `ENUM_ALL`
-----
**function** ``\texttt{ENUM\_ALL}````(````\texttt{Xs}``, ``\texttt{idx}``, ``\texttt{assignment}````)``

``\;\;`` **if** ``\texttt{idx} > \texttt{length}(\texttt{Xs})`` **then return**


``\;\;`` **for** ``x \in \texttt{Xs}[\texttt{idx}]``


``\;\;\;\;\;\;`` ``\texttt{assignment} \leftarrow \texttt{assignment}\;\cup\; \{X_{\texttt{idx}}=x\} ``

``\;\;\;\;\;\;`` ``\texttt{ENUM\_ALL}(\texttt{Xs}, \texttt{idx}+1, \texttt{assignment})``


``\;\;`` **end**

**end**

-----


"""

# ╔═╡ c212283d-04c5-45ba-a44a-fcbcdf8f67a7
# md"""
# ```julia
# function enum_all(cur_idx, Xs, storage = [...])
# 	# base case, when the storage is full
# 	if cur_idx > length(Xs)
# 		println(tape) # or do some computation
# 	else
# 		for each x in Xs[cur_idx]
# 			# fill the tape at cur_idx
# 			storage[cur_idx] = x
# 			# move on to the next variable recursively
# 			enum_all(cur_idx + 1, Xs, storage)
# 		end
# 	end
# end


# ```
# """

# ╔═╡ e9e3d566-267c-4384-bb79-4336f82d220d
# md"""
# ## 


# #### The **inputs**:
# * `Xs`: a list of variables to be enumerated, *e.g.* `Xs = [[0,1], [0,1], ...]`
# * `idx` = 1,2, ..., `length(Xs)`: current index of the variable to enumerate;
# * `assignment = [_, _, ..., _]`: assignments of `Xs`, e.g. a pre-allocated fix-sized array/dictionary to rewrite the combinations


# """

# ╔═╡ 28b07cb5-9043-4118-9421-4dd40f270ebc
md"""

## Demonstration
"""

# ╔═╡ f17bc87d-129e-4b45-bbe3-f97fcfaaa98d
md"Show backtrack tree: $(@bind show_tree CheckBox(default=false))"

# ╔═╡ f9b519e0-968a-4cdf-b5b2-6d81ae88b489
"""
Enumerate all combinations of variables in Xs by backtracking

#### Arguments
- `Xs::List`: a list of variables' domains, e.g. [[0,1], [0,1],[0,1]]
- `cur_idx::Int`: the current index ranging from 1 to `length(Xs)`
- `storage::Array`: a pre-allocated array to store the assignment
"""
function enum_all(Xs, cur_idx, storage)
	if cur_idx > length(Xs)
		# println(storage)	
		@info storage # or compute something interesting with the combination
	else
		for v in Xs[cur_idx]
			storage[cur_idx] = v
			enum_all(Xs, cur_idx+1, storage)
		end
	end
end

# ╔═╡ e4aa35a5-cc15-4347-b45a-333fad211ca4
begin
	num_of_vars = 3
	num_of_choices = 0:1
	Xs = [num_of_choices for _ in 1:num_of_vars]
	# Xs = [0:1, 1:3]
	# Xs = [[:🍕, :🍎, :🍔], [:🐱, :🐶, :🐔], [:🌴, :🌻]]
end;

# ╔═╡ 35a1d725-af18-4c58-ba54-fca609606986
enum_all(Xs, 1, Array{Any}(undef, length(Xs)))

# ╔═╡ d9691606-30f7-4dbb-a0da-7a434c36838e
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
		enum_all_with_tree!(Xs, 1, Array{Any}(undef, length(Xs)), graph, names)
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

# ╔═╡ 2ac92dd2-0e98-4b73-9147-644fb0771e06
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

# ╔═╡ 6dd10abf-632f-41da-8dd9-a077abd23394
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

# ╔═╡ d04334cd-53df-42a8-ab4b-a199b30f2746
# using Graphs

# ╔═╡ b3799775-d31c-4331-951e-18a1ac97b0b5
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

# ╔═╡ 99cd888c-7b28-4a5e-9b95-910825921f43
md"""

## An example
"""

# ╔═╡ 8d91ff88-e291-4365-b2a9-5bbb9fbd39bd
md"""

$$\Large\begin{align}P(b|j,m)
&\propto P(b)\sum_{e'}P(e')\sum_{a'} P(a'|b,e')P(j|a')P(m|a')\end{align}$$


* the following is the call tree for $b= +b$
"""

# ╔═╡ 4fb87b71-5e47-4183-92af-7b20b6436efa
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/enum_ask.png" width = "450"/></center>"""

# ╔═╡ e724d8ab-2e8b-408c-9623-e59eeb312c3c
md"""
## Summary: exact inference


#### `ENUM-ASK` algorithm is an exact inference algorithm

* ##### Time complexity: ``O(2^N)`` (``N`` is the size of the BN)



* ##### A *rule of thumb*: for a BN with 25+ nodes, exact inference no longer works
  * ##### we need approximate inference algorithms

"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
TikzGraphs = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"

[compat]
Distributions = "~0.25.122"
Graphs = "~1.13.1"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.10"
LogExpFunctions = "~0.3.29"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.75"
TikzGraphs = "~1.4.0"
TikzPictures = "~3.4.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "6cdabfd3187f903e00b1b2a10a150c91f23eee1f"

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

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3bc002af51045ca3b47d2e1787d6ce02e68b943a"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.122"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "5bfcd42851cf2f1b303f51525a54dc5e98d408a3"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.15.0"
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
git-tree-sha1 = "f85dac9a96a01087df6e3a749840015a0ca3817d"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.17.1+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "2c5512e11c791d1baed2049c5652441b28fc6a31"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.4+0"

[[deps.GettextRuntime_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll"]
git-tree-sha1 = "45288942190db7c5f760f59c04495064eedf9340"
uuid = "b0724c58-0f36-5564-988d-3bb0596ebc4a"
version = "0.22.4+0"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "68c173f4f449de5b438ee67ed0c9c748dc31a2ec"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.28"

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

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "0ee181ec08df7d7c911901ea38baf16f755114dc"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Logging", "Parsers", "PrecompileTools", "StructUtils", "UUIDs", "Unicode"]
git-tree-sha1 = "5b6bb73f555bc753a6153deec3717b8904f5551c"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "1.3.0"

    [deps.JSON.extensions]
    JSONArrowExt = ["ArrowTypes"]

    [deps.JSON.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "bf36f528eec6634efc60d7ec062008f171071434"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "3.0.0+1"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "eb62a3deb62fc6d8822c0c4bef73e4412419c5d8"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.8+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1c602b1127f4751facb671441ca72715cc95938a"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.3+0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c8da7e6a91781c41a863611c7e966098d783c57a"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.4.7+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "be484f5c92fad0bd8acfef35fe017900b0b73809"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.18.0+0"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3acf07f130a76f87c041cfb2ff7d7284ca67b072"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.41.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "76374b6e7f632c130e78100b166e5a48464256f8"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.4.0+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "d922b4d80d1e12c658da7785e754f4796cc1d60d"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.36"
weakdeps = ["StatsBase"]

    [deps.PDMats.extensions]
    StatsBaseExt = "StatsBase"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "db76b1ecd5e9715f3d043cec13b2ec93ce015d53"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.44.2+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

[[deps.Poppler_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Pkg", "libpng_jll"]
git-tree-sha1 = "e11443687ac151ac6ef6699eb75f964bed8e1faa"
uuid = "9c32591e-4766-534b-9725-b71a8799265b"
version = "0.87.0+2"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "07a921781cab75691315adc645096ed5e370cb77"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.3.3"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "9da16da70037ba9d701192e27befedefb91ec284"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.2"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "5b3d50eb374cea306873b371d3f8d3915a018f0b"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.9.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "f2685b435df2613e25fc10ad8c26dddb8640f547"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.6.1"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

    [deps.StaticArrays.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6ab403037779dae8c514bad259f32a447262455a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.4"

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
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "064b532283c97daae49e544bb9cb413c26511f8c"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.8"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "91f091a8716a6bb38417a6e6f274602a19aaa685"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.5.2"

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

    [deps.StatsFuns.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.StructUtils]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "79529b493a44927dd5b13dde1c7ce957c2d049e4"
uuid = "ec057cc2-7a8d-4b58-b3b3-92acb9f63b42"
version = "2.6.0"

    [deps.StructUtils.extensions]
    StructUtilsMeasurementsExt = ["Measurements"]
    StructUtilsTablesExt = ["Tables"]

    [deps.StructUtils.weakdeps]
    Measurements = "eff96d63-e80a-5855-80a2-b1b0885c5ab7"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.8.3+2"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Tectonic]]
deps = ["Pkg"]
git-tree-sha1 = "0b3881685ddb3ab066159b2ce294dc54fcf3b9ee"
uuid = "9ac5f52a-99c6-489f-af81-462ef484790f"
version = "0.8.0"

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
deps = ["LaTeXStrings", "Poppler_jll", "Requires", "Tectonic"]
git-tree-sha1 = "4e75374d207fefb21105074100034236fceed7cb"
uuid = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"
version = "3.4.2"

[[deps.Tricks]]
git-tree-sha1 = "311349fd1c93a31f783f977a71e8b062a57d4101"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.13"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "b5899b25d17bf1889d25906fb9deed5da0c15b3b"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.12+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aa1261ebbac3ccc8d16558ae6799524c450ed16b"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.13+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "52858d64353db33a56e13c341d7bf44cd0d7b309"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.6+0"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a4c0ee07ad36bf8bbce1c3bb52d21fb1e0b987fb"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.7+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "7ed9347888fac59a618302ee38216dd0379c480d"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.12+0"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXau_jll", "Xorg_libXdmcp_jll"]
git-tree-sha1 = "bfcaf7ec088eaba362093393fe11aa141fa15422"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.1+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a63799ff68005991f9d9491b6e95bd3478d783cb"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.6.0+0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "446b23e73536f84e8037f5dce465e92275f6a308"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.7+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "5cb3c5d039f880c0b3075803c8bf45cb95ae1e91"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.51+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"
"""

# ╔═╡ Cell order:
# ╟─3a68dad0-0016-4316-8846-ccb5425658ad
# ╟─8e953cf6-e9bd-4ab7-b332-1706824c6e91
# ╟─34c34925-daed-42fe-bdc2-1cf040fd4711
# ╟─c28f13a8-c2c5-4f50-b092-33ec1f7d7516
# ╟─f265348c-e69f-499d-8ebb-634416d6b14e
# ╟─96c1556d-b398-486e-a920-791426c8a09a
# ╟─f1fe3dd3-98de-4288-b1d8-affd3d190584
# ╟─0033eac7-c9dd-43af-b83d-25c99dca82e4
# ╟─94c0287f-29f9-437f-b252-c8c903531962
# ╟─2a50a817-9594-4bd3-a45f-5d9a69a2ec6b
# ╟─5768a0f8-3730-4d4c-a63f-00da4bc96567
# ╟─c56947cb-0b9f-4b64-94e0-cf2e9e9b74dc
# ╟─543a9a65-29ea-486e-b51f-9c5a7c4a0656
# ╟─8af7d7a2-0d16-43ef-9bb4-5ab12b0cfdef
# ╟─ee52fb06-eeb7-481b-b330-4099a4205d0e
# ╟─dbe7ace2-f9be-44ad-aff6-3c10fc9861d9
# ╟─1eb5e94d-414a-4a90-bdfa-cf14e02150dd
# ╟─f70b2cbb-5575-49f6-8ecf-aff40d5479c2
# ╟─ddfb8066-258d-43e7-b024-600d3df08c35
# ╟─f9053f31-663c-4b12-a4b2-768f64f271e1
# ╟─c321eeee-2a4b-45a3-a40e-3a32aaa35cd8
# ╟─e9f05552-2173-4d73-897c-71ee41b746d2
# ╟─b68f0f17-9d28-4b90-a519-c76853098976
# ╟─83f2d335-2df3-411e-a85a-48a37755120c
# ╟─d292ee9c-49d9-443a-ae64-42e485ed7825
# ╟─2b828f8f-1e1a-4ac0-b4d6-c8e2ef64adba
# ╟─366d22ca-e42d-48fb-8292-b7a5f8335ffe
# ╟─cf82fecc-ee7c-41a5-abb5-7ecb161333a6
# ╟─f09f12ad-a97a-4f80-88e7-1173a33b7046
# ╟─75b3cecc-8729-4b67-b4b8-9b251e0cf6dd
# ╟─a749ec19-ed6c-42ea-a6cd-d576f024e0cb
# ╟─9ef1dd1a-7d31-43c2-8d2a-57e4325bd92f
# ╟─e23b5d46-33e5-4a08-81aa-0cde9b551263
# ╟─74837ad6-a744-4758-9182-e1d4b883ed70
# ╟─bab2725c-8ed1-44f2-b07c-9318d2d91385
# ╟─c627354d-b66b-4525-8651-a831be1d237f
# ╟─e287205c-565d-4b05-8ad4-4f3990081bbf
# ╟─e84fbb5e-b289-4ae5-b8bc-554789a37573
# ╟─8fcce286-6a13-4238-925e-c7afe764c59d
# ╟─dfdd07d7-121f-4dd4-b26f-273c6e4a688d
# ╟─2b2d21b5-6028-4c36-905a-f7cea4ae5ef1
# ╟─7eaaa6ef-f19d-405c-8f00-172d48972ae9
# ╟─5dcfcea3-ee02-42ff-bff4-a0182feb18e7
# ╟─b104c8b1-c4f9-4ac8-b25c-1aa7d4664d48
# ╟─f08a2a91-6ff8-4d17-93c0-620d779418c4
# ╟─97a9c33b-94bf-44ea-b510-057a5e9b9be4
# ╟─6446ffaf-ce4f-4915-8121-ad5a7855b7f2
# ╟─029e24d2-8524-4d3e-b6af-0dc76411965c
# ╟─c3110f09-d152-4bf6-a08f-d6dd79538805
# ╟─0d163494-7c51-44bc-865b-436b0f3c6259
# ╟─14505638-8a59-4a8d-9799-ef86ea306e8d
# ╟─c212283d-04c5-45ba-a44a-fcbcdf8f67a7
# ╟─e9e3d566-267c-4384-bb79-4336f82d220d
# ╟─28b07cb5-9043-4118-9421-4dd40f270ebc
# ╟─f17bc87d-129e-4b45-bbe3-f97fcfaaa98d
# ╠═f9b519e0-968a-4cdf-b5b2-6d81ae88b489
# ╠═e4aa35a5-cc15-4347-b45a-333fad211ca4
# ╠═35a1d725-af18-4c58-ba54-fca609606986
# ╟─9428adb1-48f6-49d1-9961-2fd7e95127ce
# ╟─d9691606-30f7-4dbb-a0da-7a434c36838e
# ╟─2ac92dd2-0e98-4b73-9147-644fb0771e06
# ╟─6dd10abf-632f-41da-8dd9-a077abd23394
# ╟─d04334cd-53df-42a8-ab4b-a199b30f2746
# ╟─b3799775-d31c-4331-951e-18a1ac97b0b5
# ╟─99cd888c-7b28-4a5e-9b95-910825921f43
# ╟─8d91ff88-e291-4365-b2a9-5bbb9fbd39bd
# ╟─4fb87b71-5e47-4183-92af-7b20b6436efa
# ╟─e724d8ab-2e8b-408c-9623-e59eeb312c3c
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
