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

# ╔═╡ 2dd6ec04-37fd-11ed-147f-5bc50a90e4bb
begin
	using PlutoTeachingTools
	using PlutoUI
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using Distributions
	# using LinearAlgebra
	using StatsPlots
	using LogExpFunctions
	# using Statistics
	using StatsBase
	using LaTeXStrings
	using Latexify
	using Random
	# using Dagitty
end

# ╔═╡ 03c37940-c5d7-4b9d-85bc-2427bb89c5e5
begin
	using TikzGraphs
	using Graphs
	using TikzPictures # this is required for saving
end;

# ╔═╡ eeb9e3c7-974d-4f30-b960-f31b952cb79d
TableOfContents()

# ╔═╡ 4e1ddb4c-eea9-4bdf-8183-8addc3c38891
ChooseDisplayMode()

# ╔═╡ 92072c1d-9ee7-4d99-826e-146427d0d85d
md"""


# CS3105 Artificial Intelligence


### Uncertainty 4
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 0120c238-e093-41e2-b2e0-53383a1e4880
# md"""
# # Quick quiz -- probability theory

# ## A quick quiz

# !!! quiz "Quiz 1"
# 	###### Which of the following is (are) correct?

#     * ``P(X, Y, Z|W) \propto P(X, Y, Z, W)``
# 	* ``P(X, Y|Z, W) \propto P(X, Y, Z, W)``
# 	* ``P(X|Y, Z, W) \propto P(X, Y, Z, W)``


# !!! hint "Answer"

# 	First of all, what ``P(X|Y) \propto P(X, Y)`` really implies is 
# 	```math
# 		P(X|Y=y) \propto P(X, Y=y), \text{for all }y
# 	```
# 	that is, the conditional ``P(X|Y=y)`` is proportional to the corresponding a "slice" of the joint ``P(X, Y=y)`` (or the `select` operation)


# 	**All are correct**! Remember random vectors are random variables. Just treat the ensembles as one unit, and apply conditional probability rules. 

# 	For example, the second one, we can treat ``\{X,Y\}=S_1``, and ``\{Z,W\} = S_2``
# 	then it is obvious ``P(S_1|S_2) \propto P(S_1, S_2)``; therefore ``P(X,Y|Z=z,W=w) \propto P(X,Y,Z=z,W=w).`` It still means the conditional is proportional to a *slice* of the joint but the slice is a "2-d array" and sliced with two conditions.


# """

# ╔═╡ 03701ff0-0166-4eed-a326-facd51450cae
# aside(tip(md"""
# Also remember ``\propto`` relationship is reflexive and transitive, that is if ``x\propto y``, then ``y\propto x``; and if ``x \propto y, y \propto z``, then ``x \propto z``
# * it is easy to prove it, if ``x\propto y``, then ``x = c\cdot y`` for some ``c\neq 0``, which implies ``y = \frac{1}{c} x``, which is the exact definition of ``y\propto x``.
# * the transitivity is also obvious and left as an exercise

# """))

# ╔═╡ 43a70396-26bc-4d24-b17b-5ec3daaccad5
# md"""



# ## A quick quiz (2)

# !!! quiz "Quiz 2"
# 	###### Are the following statements correct and why? 
    
# 	* ``P(X | Y, Z) \propto P(X) P(Y) P(Z)``
# 	* ``P(X|Y, Z) \propto P(X, Y | Z)``
# 	* ``P(X|Y) \propto P(Y)P(X|Y)``


# !!! hint "Answer"
# 	* the first is wrong! ``P(X|Y,Z) \propto P(X,Y,Z)`` but ``P(X,Y,Z) \neq P(X)P(Y)P(Z)`` in general (unless independence is assumed)

#     * the second is correct; there are multiple ways to show it. One approach is to remember conditional probability distributions are valid distributions. Therefore, it is just conditioned version of chain rule. Alternatively, both sides proportional to the joint; ``\propto`` is transitive. ``P(X|Y,Z)\propto P(X,Y,Z)`` and ``P(X,Y|Z) \propto P(X,Y,Z)``; then ``P(X|Y,Z)\propto P(X,Y,Z) \propto P(X,Y|Z)``.

# 	* correct; ``P(X|Y) \propto P(X,Y) = P(Y)P(X|Y)`` just chain rule.

# """

# ╔═╡ 9211c8ac-43b7-43e3-846c-f3764e79706c
md"""

# Recap




## Bayesian network

##### A Bayesian network is defined as

$\large \text{bn} = \{G, \{P\}\}$

* #### ``G``: _a DAG_
  * ##### one random variable per node 
\


* ``\Big\{P(X_i|\text{parent}(X_i))\Big\}`` for ``i=1,\ldots, n`` the CPFs
  * ##### one CPF for each node



"""

# ╔═╡ c74cef62-a137-45e2-960c-dcc0944810fb
html""" <center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglarCPTs.png" width = "700"/></center>
"""

# ╔═╡ 95b544dd-d3b4-4c64-8269-9a0bf1013b54
md"""
## Bayesian network: factoring property

!!! important "Factoring property"
	
	$\Large P(X_1, X_2,\ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))$
	* ##### a Bayes' net corresponds to a full probabilistic model (joint distribution)
"""

# ╔═╡ ca80d078-c859-4cb0-8f5b-ff4ec88848aa
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

# ╔═╡ d8c17f75-000c-4999-9aed-f2f92d6edeb2
md"""

# Today

\


* ####  More Bayes' network examples


* #### Bayes' network in machine learning
  * ##### Bayesian logistic regression

"""

# ╔═╡ 166c832c-a4ff-470e-818c-e66ccbba2fac
md"""
# Bayes' net examples


## Bayes' network's construction

\


#### A typical approach is to follow cause-effect order


* ##### cause first 


* ##### then link effect from causes
"""

# ╔═╡ 1fae859c-bdfa-41d1-8b8b-f066cb6d9ee9
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "350"/></center>""" 

# ╔═╡ f2172897-1f13-49ed-a5ce-07adf3e61899
md"""

## Examples with bad orders


"""

# ╔═╡ 5a912a9a-1f4d-4841-9070-3c866b43aae2

md"""
##### All BNs lead to the *same joint distribution* ``P(B,E,A,M, J)``
* numbers next to each node are the required parameters for the CPT

"""

# ╔═╡ 26a21968-819c-4cb1-9c71-16f54599b7de
TwoColumn(md"""
> \[`MaryCalls`, `JohnCalls`, `Alarm`, `Burglar`, `Earthquake`\]
* in total 13 parameters 

""", md"""

> \[`MaryCalls`, `JohnCalls`, `Earthquake`, `Burglar`, `Alarm` \]
* in total 31 parameters!
* nodes are linked all ancesters
""")

# ╔═╡ 4e0adc00-d644-4290-8b26-c034945992f7
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/wrongorders.png" width = "800" /></center>
"""

# ╔═╡ a8be829e-81d9-437c-8df1-2673ff3233e4
# md"""


# !!! conclusion "Conclusion"
# 	Bad order leads to overly complicated models
#       * less compact: the original model has only 10 parameters
#       * much harder to specify or learn the CPTs as well!
# """

# ╔═╡ 8da3fb38-3ed4-494c-be8b-e8ff92e8bf90
md"""

## Coin guess example


!!! problem "Coin guess"
	#### Two coins (with different biases) in an urn
	* #####  Coin 1 is fair: $p_{1}= 0.5$
	* ##### Coin 2 is very bent: $p_2=0.01$: very unlikely to observe ``\texttt{head}``
	
	##### Randomly pick one coin from the urn and toss it 2 times and find out

	$$\Large Y_1, Y_2\in \{\texttt{h},\texttt{t}\}$$

	##### *Which coin* has been used ? If we toss again, what is ``Y_3``?

"""

# ╔═╡ 6d3a0b93-6d11-4a21-9618-245b8d7e0816
md"""

## Coin guess example 

#### (step 0: identify random variables)

\


#### *Identify* the random variables 

$$\Large C,\; Y_1,\; Y_2, \; Y_3$$ 

  * ##### ``C= 1,2`` denote which coin has been used 


  * ##### ``Y_1, Y_2, Y_3`` are the tossing outcomes (``Y_3`` for future toss/prediction)

"""

# ╔═╡ 160853f2-35fd-4219-9d7a-94baee003888
md"""

## Coin guess example 

#### (step 1: order random variables)

\


#### Think **causally or generatively**:

```math
\Large C \Rightarrow Y_1, Y_2, Y_3
```


"""

# ╔═╡ fb9a604e-2a79-4832-8eab-c2f251f9947f
# html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/diceExample0.png" width = "400"/></center>
# """

# ╔═╡ aaf0aad5-97c7-4ee8-9081-f078a14a6aec
md"""

## Coin guess example 

#### Aside: no conditional independence
\



#### *Without* conditional independence assumption (or by chain rule)

$\Large P(C, Y_1, Y_2,Y_3) = P(C)P(Y_1|C)P(Y_2|C, Y_1)P(Y_3|C, Y_1, Y_2)$

"""

# ╔═╡ d13c1452-56db-4d3e-9401-59797d5a8ffc
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/coin_full.png" width = "400"/></center>"""

# ╔═╡ 5856496b-a0c2-48d6-874e-80b46b0297d7
md"""

## Coin guess example 

#### (Step 2: trim the edges)
\


#### ``Y_2``: consider [``\texttt{Coin Choice}``, ``\cancel{\texttt{Y1}}``]
* ``Y_2`` only depends on _`Coin Choice`_ (common cause pattern)

#### ``Y_3``: consider [``\texttt{Coin Choice}``, ``\cancel{\texttt{Y1}}``, ``\cancel{\texttt{Y2}}``]
* ``Y_3`` only depends on _`Coin Choice`_ (common cause pattern again)

"""

# ╔═╡ 2494ef53-fc58-4152-82f2-8334778eb3c9
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/coin_choice.png" width = "400"/></center>"""

# ╔═╡ 1d46065d-318e-49d7-b9e2-16fff9222a68
md"""

#### In other words, the BN

$\large P(C, Y_1, Y_2,Y_3) = P(C)P(Y_1|C)P(Y_2|C, \cancel{Y_1})P(Y_3|C, \cancel{Y_1, Y_2})$


"""

# ╔═╡ c355a185-e35a-429b-a359-4356623a5f22
md"""
## Coin guess example 
#### (Step 3: add CPFs)



"""

# ╔═╡ b5a21bae-66cf-4c07-aa67-69987c7cd459
html"""

<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/coinguessdag.png" width = "600"/></center>

"""

# ╔═╡ 5209f762-3eb6-432a-8e4c-a0aa2cde833c
md"""

## Plate notation 


#### What if we toss the coin **100** times ?

* drawing 100 nodes seems tedious and unnecessary as well



#### *Plate* notation is invented for repeated process
  * just like a `for` or `while` loop


"""

# ╔═╡ 962da2f3-4a9b-4cf4-80af-fac1253231af
html"""

<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/coinguessdagplate.png" width = "200"/></center>

"""

# ╔═╡ 9d61b845-017d-4db6-a51a-c52a85747de5
md"""

# Probabilistic inference: exact inference
"""

# ╔═╡ 6a4fa5f4-7e21-4528-8792-98a1a6ce8b82
md"""

## Probabilistic inference 

!!! infor "Probabilistic Inference"
	```math
	\Large P(Q|E_1=e_1, E_2= e_2, \ldots, E_k= e_k)
	```

	* ``Q``: **query** random variable
	* ``E_i``: **evidence** (evidence can be empty)
    * ``N_i``: all the others: **nuisance** random variables 
"""

# ╔═╡ 9a19c0d8-6ce6-4597-ae10-665aa088b820
md"""

## Inference based on Bayes' net



"""

# ╔═╡ 4e883d98-8f75-4bcf-b2d2-2d06e8595c91
TwoColumnWideLeft(

md"""
Some inference examples are 

###### ``P(J)``: _how likely John calls ?_
* query: ``J``
* evidence: ``\emptyset``
* nuisance: ``B,E,A,M``


###### ``P(J|+b)``: _how likely John calls if Burglary happens?_
* query: ``J``
* evidence: ``B=+b``
* nuisance: ``E, A`` and ``M``



###### ``P(B|+j,+m)``: _how likely Burglary happens given both John and Mary calls?_
* query: ``B``
* evidence: ``J=+j, M =+m``
* nuisance: ``E`` and ``A``

""",

	
html"""<br><br><br><br><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "300"/></center>"""
	
)

# ╔═╡ 252602ae-c223-43da-a3fa-8a59dfc02987
md"""


## Query types


* **bottom-up**: given evidence (observations, data) infer the cause; opposite the direction of edges, 
  * *e.g.* $P(B|J,M)$
* **top-down**: given cause infer downstream r.v.s (follow the direction of edge), 
  * prediction of future data
  * *e.g.*  $P(J|B, E)$

* **mixture of both**: $P(J|M)$

"""

# ╔═╡ e1657fab-1783-4ad6-b292-d7eab019341d
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglar.png" width = "300"/></center>"""	

# ╔═╡ af69eea3-8e46-4b22-8d27-c444ac41f090
md"""

## Example 

##### Let's consider 


$\large P(J|+b, -e)$


!!! question ""
	**By conditional probability definition**
	```math
	P(J|+b, -e) = \frac{P(J, +b, -e)}{P(+b, -e)} = \frac{P(J, +b, -e)}{P(+j, +b, -e) + P(-j, +b, -e)}
	```
##

Note that the denominator (sum rule) is a **normalising constant**:

```math
P(+b, -e) = P(+j, +b, -e) + P(-j, +b, -e)\tag{sum rule}
```

*  and it can be computed from the numerator ``P(J, +b, -e)`` 


##

!!! question ""
	Therefore, queries are often written as  

	$$P(J|+b, -e) \propto P(J, +b, -e)$$ 

	or 
	
	$$P(J|+b, -e) =\alpha P(J, +b, -e)$$ 
	* where ``\alpha = \frac{1}{P(+b, -e)}`` is the normalising constant


"""

# ╔═╡ b758d849-44a1-48ca-8c1b-9066965d83a6
md"""
## Example: cont.

##### Next, by ``\textcolor{red}{\text{sum rule}}`` 
* sum out the nuisance random variables

!!! question ""
	$$\begin{align}
	P(J|+b, -e) &= \alpha P(J, +b, -e)\\
	\Aboxed{&= \alpha \sum_{a'}\sum_{m'} P(+b,-e,a',J,m')\;\;\;\textcolor{red}{\text{sum rule}}}
	\end{align}$$



"""

# ╔═╡ 95dd0d16-00b4-455e-899d-1ab93f035946
md"""
## Example: cont.

##### Next, by ``\textcolor{blue}{\text{factoring property}}`` of the BN

!!! question ""
	$$\begin{align}
	P(J|+b, -e) &= \alpha P(J, +b, -e)\\
	&= \alpha \sum_{a'}\sum_{m'} P(+b,-e,a',J,m')\;\;\;\textcolor{red}{\text{sum rule}}\\
	\Aboxed{&=\alpha \sum_{a'}\sum_{m'} P(+b)P(-e)P(a'|+b,-e)P(J|a')P(m'|a')\;\;\;\textcolor{blue}{\text{factor property}}} 
	\end{align}$$

"""

# ╔═╡ 324d4b06-fe82-472d-9153-cb0665234d9b
md"""


##

As a concrete example, to find, *e.g.*

$\large P(J|+b, -e)$


##### For $J=+j$ : 

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

##### For $J=-j$ : 

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

**Lastly,** normalise to find 
$\alpha = \frac{1}{0.0008473+ 0.0001507}$

$$\begin{align}
P(J|+b, -e) &\propto \begin{cases} 0.00847302 & J= +j \\ 0.0001507 & J=-j \end{cases}\\
&= \begin{cases} 0.849 & J= +j \\ 0.151 & J=-j \end{cases}
\end{align}$$


"""

# ╔═╡ 40d78c6c-bf36-48b8-abef-02721b57d305
md"""

!!! question "Question"
	###### How many _multiplications_ involved?
"""

# ╔═╡ 76d385de-4eb4-4b4b-aaf3-659b6a31691a
Foldable("", md"
Roughly in the order

```math
2^{N_k} \times \text{\# of factors} 
```

* ``N_k``: the number of nuisance variables
* ``\#`` of factors: or number of nodes in a BN
* ``O(2^{N_k}\cdot N)`` complexity
")

# ╔═╡ 4f32a716-6be3-4aa6-8745-c586ac80e248
# html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/burglarCPTs.png" width = "600"/></center>""" 

# ╔═╡ b6999f5f-c92a-4a30-bf32-336604490512
md"""

## Bottom-up case: ``P(B| J, M)``

###### For query ``P(B|J=j, M=m)``, the nuisance random variables: ``\{E,A\}``


$$\begin{align}P(B&|J=j,M=m) = \alpha P(B,J=j,M=m)\;\;\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\text{:conditional probability} \\
&= \alpha \sum_{e'}\sum_{a'} P(B,e',a',J=j,M=m)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\,\;\;\;\;\;\;\;\;\;\; \text{:sum rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B)P(e')P(a'|B,e')P(J=j|a')P(M=m|a') \;\; \text{:factor property}\end{align}$$


* for each ``B=b``, `enumerate` all ``(e', a') \in \{\texttt{t}, \texttt{f}\}^2``, then `sum`, 
* then `normalise`
"""

# ╔═╡ 6d936c43-1c14-4cb9-b7fe-b826bf4f6a8d
md"""
## Mixture of both ``P(J|M=m)``



###### For query ``P(J|M=m)``, nuisance random variables: $\{B,E, A\}$

$$
\begin{align}
P(J|M&=m) = \alpha P(J, M=m)\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;\;\, \;\;\;\;\;\; \;\;\;\;\;\,\;\;\;\;\;\;\;\;\;\;\;\; \;\;\;\;\;  \color{blue}\text{:conditional probability}\\
&= \alpha \sum_{b'}\sum_{e'}\sum_{a'} P(B=b',E=e',A=a',J,M=m)\;\;\, \;\;\;\;\;\;\;\color{blue}\text{:sum rule}\\
&=\alpha \sum_{b'}\sum_{e'}\sum_{a'}  P(b')P(e')P(a'|b',e')P(J|a')P(M=m|a')\,\color{blue}\text{:factoring property}
\end{align}$$

Specifically,

* for each ``J=j``, `enumerate` ``(b', e', a')\in \{\texttt{t,f}\}^3``, then `sum`, 
* then `normalise`
	
"""

# ╔═╡ b2147c2b-7677-4c51-9cd1-1823b09e930e
md"""

## All queries are the same


!!! information ""
	Given a BN and query ``P(Q|\mathcal{E}=\mathbf{e})``:
	```math
	\begin{align}
	P(Q|\mathcal{E}=\mathbf{e}) &\propto P(Q, \mathcal{E}=\mathbf{e}) \\
	&= \sum_{n_1}\ldots \sum_{n_k} P(Q, \mathcal{E}=\mathbf{e}, n_1, n_2\ldots n_k)\\
	&= \sum_{n_1}\ldots \sum_{n_k} \left ( \prod_i \text{CPFs}\right )
	\end{align}


	```
	* ``Q``: query random variable 
	* ``\mathcal{E}``: conditioned evidence 
	* ``\mathcal{N}``: nuisance (or hidden) random variables



* ###### more nuisance random variables ``\Rightarrow`` more sums



"""

# ╔═╡ 37320389-2c6d-4e18-a51e-6110d646d543
aside(tip(md"""

``\mathcal{N, Q, E}``: this font denotes sets of random variables

"""))

# ╔═╡ a2bdbcea-372a-49d1-b3ac-863b16693afd
md"""

# Inference algorithm



#### So far, we do it by hand
\
\

#### _How to do it with computers (a.k.a. an algorithm)? *_

"""

# ╔═╡ 1d9f5511-2aca-49ce-b413-11a4521a9222
md"""

## A naive inference algorithm



------
**A naive exact inference algorithm**

**Step 0.** tabulate and store the full joint distribution based on 

```math
P(X_1, \ldots, X_n) = \prod_{i=1}^n P(X_i|\text{parent}(X_i))
```


**Step 1.** for each value ``q`` that ``Q`` can take (*i.e.* apply sum rule)

   * find rows matches ``Q=q \wedge \mathcal{E} =\mathbf e``
   * sum their probabilities and store in ``\texttt{vec}[q]``


**Step 2.** normalise ``\texttt{vec}`` then return it
  
$$P(Q=q|E=\mathbf{e}) = \frac{\texttt{vec}[q]}{\sum_q \texttt{vec}[q]}$$

------	
"""

# ╔═╡ 7f40c895-ea88-4308-9dad-4ae910082160
md"""
## A naive inference algorithm

For example, query 

$P(J|B=\texttt{t},E=\texttt{t})$

**Step 0**: populating the full joint table:


"""

# ╔═╡ 01c7569e-c635-4c47-a882-4c55cc33f79e
html"
<table>
<thead>
    <tr>
        <td>B</td>
        <td>E</td>
        <td>A</td>
        <td>J</td>
        <td>M</td>
		<td>P(B,E,A,J,M)</td>
    </tr>
  </thead>
    <tr>
        <td>f</td>
        <td>f</td>
        <td>f</td>
        <td>f</td>
        <td>f</td>
		<td>..</td>
    </tr>
	<tr>
        <td>f</td>
        <td>f</td>
        <td>f</td>
        <td>f</td>
        <td>t</td>
		<td>..</td>
    </tr>

	<tr>
        <td>f</td>
        <td>f</td>
        <td>f</td>
        <td>t</td>
        <td>f</td>
		<td>..</td>
    </tr>

	<tr>
        <td>f</td>
        <td>f</td>
        <td>t</td>
        <td>t</td>
        <td>t</td>
<td>..</td>
    </tr>

	<tr>
        <td>f</td>
        <td>t</td>
        <td>t</td>
        <td>f</td>
        <td>f</td>
<td>..</td>
    </tr>

	<tr>
		<td> ⋮</td> 
        <td>⋮</td>
        <td>⋮</td>
        <td>⋮</td>
        <td>⋮</td>
        <td>⋮</td>
    </tr>

	<tr>
        <td>t</td>
        <td>t</td>
        <td>t</td>
        <td>t</td>
        <td>t</td>
<td>..</td>
    </tr>
</table>"

# ╔═╡ 7a4bfa6b-f485-4074-b03c-358f6e66102f
md"""

**Step 1**: enumerate query variable ``J`` and apply sum rule

* for ``J=\texttt{t}``, `filter` the table with the condition ``\{J=\texttt{t} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
  * sum and store it in ``\texttt{vec}[J=\texttt{t}]``
* for ``J=\texttt{f}``, `filter` the table with the condition ``\{J=\texttt{f} \}\wedge\underbrace{\{ B=\texttt t\} \wedge \{E= \texttt t\}}_{\text{evidence: } \mathcal{E} =\mathbf{e}}``
  * sum and store it in ``\texttt{vec}[J=\texttt{f}]``

**Step 2** normalise the ``\texttt{vec}`` and return it
"""

# ╔═╡ 5dbd5b65-7710-41d4-a7f5-7df1fe4c780c
(tip(md"""

What **step 1** essentially does (sum out the nuisances):

$$\begin{align}&\texttt{vec}[J=t] =\sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=t, M=m')\end{align}$$


$$\begin{align}&\texttt{vec}[J=f] = \sum_{a'}\sum_{m'}P(B=t, E=t, A=a', J=f, M=m')\end{align}$$

"""))

# ╔═╡ f8549e05-bb5d-43fa-846f-434d0dc5d434
md"""

## Naive algorithm does not scale
"""

# ╔═╡ 6d0a9997-449b-44fd-b596-8d5533fbd77b
md"""

##### *Step 0* is not feasible !
\


###### The joint full table is **HUGE** 

* ``2^N`` rows, ``N`` is the number of random variables in the BN
* space complexity is ``O(2^{N})``


The rest two steps, however, are plainly _simple and efficient_
* just basic database `filter` operation and summation
"""

# ╔═╡ 6ffd8651-e50b-4214-a04f-df5fe8ab1464
md"""

## Digress: Enumerate all - how ?


##### To enumerate ``(x_1, x_2) \in (\texttt{t}, \texttt{f})^2``
* a nested `for` loop works

```julia
for x₁ in X₁ # assume X₁ returns the possible value x₁ can take
	for x₂ in X₂
		print(x₁, x₂) # or compute something interesting
	end
end
```


##### For three ``X_1, X_2, X_3``, 
* obviously, nest 3 `for` loops

```julia
for x₁ in X₁ # assume X₁ returns the possible value x₁ can take
	for x₂ in X₂
		for x₃ in X₃
			print(x₁, x₂, x₃) # or compute something interesting
		end
	end
end
```

##

!!! question "Question"
	What if the list ``\{X_1, X_2, \ldots, X_n\}``'s size is not known before-hand
    * ######  ``n`` is not known or fixed
    * how to enumerate all variables from a **dynamic** list

If one is desperate enough, could write a nested loop for each ``n\geq 1``...
* still will not work for ``n+1``
"""

# ╔═╡ 4cdecd2c-cb2a-4c32-8e94-dd20d878d990
md"""

## Digress: Enumerate all - Backtracking



The standard: **recursive backtracking!**



"""

# ╔═╡ 9de153f8-6874-4f9d-8fcb-9ccdc5ed9854
md"""

The **inputs**:

* `cur_idx` = 1,2, ..., `length(Xs)`: current index to track which `Xs[cur_idx]` to enumerate; I assume index starts from 1
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

# ╔═╡ ecdaabd0-66b6-40c0-b485-25e14e814115
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/back_track.png" width = "650"/></center>"""

# ╔═╡ 451cc71b-6dbd-4cc5-957d-78a4fffd1913
md"""

## Demonstration
"""

# ╔═╡ 8abeb09c-3246-429f-b818-fd6e16d4d862
md"Show backtrack tree: $(@bind show_tree CheckBox(default=false))"

# ╔═╡ ef37360b-ee56-4727-ace9-8923c1ef98ac
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

# ╔═╡ 5470b98b-0a67-435c-90dd-484c97190663
begin
	num_of_vars = 3
	num_of_choices = 0:1
	Xs = [num_of_choices for _ in 1:num_of_vars]
	# Xs = [0:1, 1:3]
end;

# ╔═╡ 3edbd822-772d-4e49-a96c-d46844ddbf15
enum_all(Xs, 1, Array{Int}(undef, length(Xs)))

# ╔═╡ 41acf75a-50da-41ed-8693-1f7426e48580
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

# ╔═╡ 560cc996-3059-48b4-a674-9b9bcc18f48e
# md"""

# ## Exercise


# !!! exercise "Exercise"
# 	Modify the backtracking method such that it returns the number of combinations such that number of 1s is an even number;

# 	*e.g.* combination of [0, 1, 1] has 2 ones; and it is a even number  
# """

# ╔═╡ 40cca076-c520-492b-adc9-4f4cf7c342fc
# function enum_all_count(Xs, cur_idx, storage)
# 	if cur_idx > length(Xs)
# 		return 1
# 	else
# 		c = 0
# 		for v in Xs[cur_idx]
# 			storage[cur_idx] = v
# 			c += enum_all_count(Xs, cur_idx+1, storage)
# 		end
# 		return c
# 	end
# end

# ╔═╡ 6098f85c-43b4-4298-8ba3-abf390a72cc9
# enum_all_count(Xs, 1, Array{Int}(undef, length(Xs)))

# ╔═╡ 8142d3bb-df1f-4865-9e98-fbb435199cf8
md"""
## A second attempt

###### Form the full joint table ``P(X_1, X_2, \ldots, X_n)`` in one-go is *neither*
* **necessary**
* nor **practical**


###### The idea of second attempt:
* use recursive *back-tracking* to *enumerate* the **nuisance random variables** only ``\mathcal{N}=\{N_1, N_2, \ldots N_k\}``

-----

**Step 0.** initialise an emtpy array: ``\texttt{vec}``

**Step 1.** for each value ``q`` that ``Q`` can take
  * recursively enumerate all combinations ``(n_1, n_2,\ldots, n_k)`` by backtrack and **accumulates**
$$\texttt{vec}[q]  \mathrel{+}=  P(Q=q, E=\mathbf e, N_1 = n_1,  \ldots, N_k=n_k)$$

**Step 2.** normalise ``\texttt{vec}`` then return it
  
$$P(Q=q|E=\mathbf{e}) = \frac{\texttt{vec}[q]}{\sum_q \texttt{vec}[q]}$$

------	
"""

# ╔═╡ a392d776-182d-4a84-91c9-923fb16acc6a
md"""

## A second attempt *vs* the first attempt


##### The second attempt is better 

* we do not enumerate all but a smaller subset


Recall recursive backtrack essentially builds a tree

* the first attempt creates a full tree and **stores** it somewhere in RAM

* the second method creates a much smaller sub-tree and **does not** store them

##

##### However, the worst case is still very bad

Consider the extreme case in which ``\mathcal{E} =\emptyset``, *i.e.* no evidence

```math
P(Q) \propto \underbrace{\sum_{n_1}\sum_{n_2}\ldots \sum_{n_k}}_{\texttt{N-1 of nuisance r.v.s} } P(Q, N_1 = n_1, \ldots, N_k =n_k)
```
* space complexity is fine (backtracking is efficient in space complexity, you cannot beat it)
  * we do not store the table
* but time complexity is ``O(N \times 2^N)``
  * ``2^N`` combinations, each has ``N`` floating number multiplications
"""

# ╔═╡ e6cd70fc-deb9-4698-af35-23c33ca82391
md"""
## Digress: summation notation

Recall the summation  

$\large \sum_{i=1}^n ax_i =ax_1 + a x_2 + \ldots + ax_n= a\left (\sum_{i=1}^n x_i\right )$

* when ``a`` is a common factor, 
  * it can be taken out or 
  * equivalently pushing summation $\,\Sigma_{\cdot}\,$ inwards
* it saves some floating multiplication operations (why)
  * remember floating number ``\times`` is more expensive than ``+/-``



##

##### Example: ``P(J|b, e)``
\

``P(B=b)P(E=e)`` is a common factor




$$\begin{align}

P(J|B=b,E=e)&= \alpha \sum_{a'}\sum_{m'} \underbrace{\boxed{P(b)P(e)}}_{\text{constant!}}P(A=a'|b,e)P(J|a')P(m'|a') \\
  &= \alpha P(b)P(e)\sum_{a'}\sum_{m'}{P(a'|b,e)P(J|a')}P(m'|a')
\end{align}$$

"""

# ╔═╡ e69fb839-218d-47ea-9c45-2236a3c82d0a
md"""
##


$$\begin{align}
P(J|B=b,E=e)&= \alpha \sum_{a'}\sum_{m'} \underbrace{\boxed{P(b)P(e)}}_{\text{constant!}}P(A=a'|b,e)P(J|a')P(m'|a') \\
  &= \alpha P(b)P(e)\sum_{a'}\sum_{m'}\boxed{P(a'|b,e)P(J|a')}P(m'|a')
\end{align}$$

``\boxed{P(a'|b,e)P(J|a')}`` together is also a common factor (from $m$'s perspective)
* or in other words, we push $\sum_{m'}$ inwards

$$\begin{align}
P(J|B=b,E=e)
  = \alpha P(b)P(e)\sum_{a'}\boxed{P(a'|b,e)P(J|a')}\sum_{m'}P(m'|a')\end{align}$$

"""

# ╔═╡ d22bd0dd-8db4-43bd-b3dc-c4d7ef4c30c3
md"""
## Further simplification

We can further simplify it, note that 


$$\sum_{m'}P(M=m'|a)=1$$

$$\Rightarrow P(J|b,e)
  = \alpha P(b)P(e)\sum_{a'}P(A=a'|b,e)P(J|A=a')\underbrace{\sum_{m'}P(M=m'|a')}_{1.0}$$

So we have 


$$P(J|b,e)
  = \alpha P(b)P(e)\sum_{a'}P(A=a'|b,e)P(J|A=a')$$

* 2 nested sums to 1 summation

* in general, it reduces from ``O(N \times 2^N)`` to ``O(2^N)``
* the difference is we compute while we back-track rather than wait until the end
"""

# ╔═╡ b81367c2-294b-42ef-8fd5-3157aeaf83ee
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

# ╔═╡ fc661023-20a7-414e-9e14-d01448a0f5b1
md"""

## Another example



$$\begin{align}P(B|j,m) &= \alpha P(B,j,m)\;\;\; \text{conditional probability rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B,e',a',j,m)\;\;\; \text{summation rule} \\
&= \alpha \sum_{e'}\sum_{a'} P(B)P(e')P(a'|B,e')P(j|a')P(m|a') \;\;\; \text{factoring CPTs} \\
&= \alpha P(B)\sum_{e'}P(e')\sum_{a'} P(a'|B,e')P(j|a')P(m|a')\;\;\; \text{simplify} \end{align}$$


"""

# ╔═╡ 4c532bc1-124f-4e85-ac5f-6c68a5d33943
md"""

## Enumerate-ask algorithm


The following is our version of `ENUM-ASK` algorithm

* very similar to the algorithm in the AI-AMP textbook
* rewritten to show the connection to the backtracking algorithm introduced before


##

-----
**function** ``\texttt{ENUM\_ASK}``(``{Q}``, ``\{\mathcal{E}=\mathbf{e}\}``, ``\texttt{bn}``)

``\;\;`` ``\texttt{Xs} = \texttt{bn.VARS}\;\;\;\;\;``   # *i.e.* 
 list of all r.v.s in a topological order (parents before child nodes)

``\;\;`` ``\texttt{vec} \leftarrow \texttt{an array with all zeros}\;\;\;\;\;\;`` # an array for ``P(Q|\mathbf e)``

``\;\;`` **for each**  value ``q`` of ``Q`` **do**

  * ``\texttt{vec}(q) \leftarrow \texttt{ENUM-ALL}(\texttt{Xs}, 1, \{\mathcal{E}=\mathbf{e}\} \cup \{Q=q\})``
``\;\;`` **endfor**

``\;\;`` **return** ``\texttt{NORMALISE}(\texttt{vec})``

**end**

-----

\


-----
**function** ``\texttt{ENUM\_ALL}````(````\texttt{Xs}``, ``\texttt{idx}``, ``\{\mathcal{E}=\mathbf{e}\}````)``

``\;\;`` **if** ``\texttt{idx} > \texttt{length}(\texttt{Xs})`` **then return** ``1.0``

``\;\;`` ``{X} \leftarrow \texttt{Xs}[\texttt{idx}]``

``\;\;`` **if** ``X \in \{\mathcal{E}=\mathbf{e}\}`` # if the variable is in the evidence set

``\;\;\;\;\;\;`` **return** ``P({x}| \texttt{parents}(X))\times \texttt{ENUM-ALL}(\texttt{Xs},  \texttt{idx+1}, \{\mathcal{E}=\mathbf{e}\})``

``\;\;`` **else**  ``\;\;`` # X is nuisance, apply sum rule 



``\;\;\;\;\;\;`` **return** ``\sum_x P(x |\texttt{parents}(X)) \times \texttt{ENUM-ALL}(\texttt{Xs}, \texttt{idx+1}, \{\mathcal{E}=\mathbf{e}\} \cup \{X =x\})``

``\;\;`` **end**

**end**

-----


"""

# ╔═╡ ee4f5fa0-4260-46fb-9186-15bbdfe34998
md"""

## An example
"""

# ╔═╡ e4724627-a016-4ae6-83ec-6bd8ca270e30
md"""

$$\begin{align}P(b|j,m)
&\propto P(b)\sum_{e'}P(e')\sum_{a'} P(a'|b,e')P(j|a')P(m|a')\end{align}$$

"""

# ╔═╡ 0debcf0e-7680-4183-8332-e28eef0dabb6
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/enum_ask.png" width = "450"/></center>"""

# ╔═╡ 506c8bd2-6256-407e-b48a-d7fbc73b3a38
md"""
## Summary: exact inference


`ENUM-ASK` algorithm is an exact inference algorithm

* time complexity from ``O(N\times 2^N)`` to ``O(2^N)`` (``N`` is the size of the BN)

The time complexity is still grim
* what's worse, it does not work for continuous random variables
  * sums become integrations: integration cannot be done analytically


##

> A **rule of thumb**: for a BN with 25+ nodes, exact inference no longer works
> * might take years to finish

Most uncertainty inference software do **not** even offer the exact inference option

* popular ones: `Stan`, `Turing.jl`, `PyMC`, `Tensorflow-probability`
* instead, they offer approximate inference such as **MCMC** or variational inference
"""

# ╔═╡ d24f59ee-c4ed-413b-a8e1-497b31f1fed1
md"""

# Sally Clark case




We will use Bayes' network to solve Sally Clark's case
* a case study 
* walk-through of the whole process: modelling + inference (with the help of BN)


Check the Jupyter-notebook on studres.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
Graphs = "86223c79-3864-5bf0-83f7-82e725a168b6"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
TikzGraphs = "b4f28e30-c73f-5eaf-a395-8a9db949a742"
TikzPictures = "37f6aa50-8035-52d0-81c2-5a1d08754b2d"

[compat]
Distributions = "~0.25.76"
Graphs = "~1.12.0"
LaTeXStrings = "~1.3.0"
Latexify = "~0.15.17"
LogExpFunctions = "~0.3.18"
Plots = "~1.35.3"
PlutoTeachingTools = "~0.2.3"
PlutoUI = "~0.7.43"
StatsBase = "~0.33.21"
StatsPlots = "~0.15.4"
TikzGraphs = "~1.4.0"
TikzPictures = "~3.5.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.1"
manifest_format = "2.0"
project_hash = "0d4530eb597f661d25659db3b5db27f87584d81a"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

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

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "8873e196c2eb87962a2048b3b8e08946535864a1"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+2"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodeTracking]]
deps = ["InteractiveUtils", "UUIDs"]
git-tree-sha1 = "7eee164f122511d3e4e1ebadb7956939ea7e1c77"
uuid = "da1fd8a2-8d9e-5ec2-8556-3022fb5608a2"
version = "1.3.6"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

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
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

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

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "3101c32aab536e7a27b1763c0797dba151b899ad"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.113"

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
git-tree-sha1 = "8e9441ee83492030ace98f9789a654a6d0b1f643"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "cc5231d52eb1771251fbd37171dbc408bcc8a1b6"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+0"

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

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4d81ed14783ec49ce9f2e168208a12ce1815aa25"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+1"

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
git-tree-sha1 = "db16beca600632c95fc8aca29890d83788dd8b23"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.13.96+0"

[[deps.Formatting]]
deps = ["Logging", "Printf"]
git-tree-sha1 = "fb409abab2caf118986fc597ba84b50cbaf00b87"
uuid = "59287772-0a20-5a39-b81b-1366585eb4c0"
version = "0.4.3"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "fa8e19f44de37e225aa0f1695bc223b05ed51fb4"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+0"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1ed150b39aebcc805c26b93a8d0122c940f64ce2"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.14+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Base64", "DelimitedFiles", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Pkg", "Preferences", "Printf", "Random", "Serialization", "Sockets", "Test", "UUIDs"]
git-tree-sha1 = "00a9d4abadc05b9476e937a5557fcce476b9e547"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.69.5"

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
git-tree-sha1 = "b36c7e110080ae48fdef61b0c31e6b17ada23b33"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.2+0"

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
git-tree-sha1 = "ae350b8225575cc3ea385d4131c81594f86dfe4f"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.12"

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
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "be3dc50a92e5a386872a493a10050136d4703f9b"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.6.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "25ee0be4d43d0269027024d75a24c24d6c6e590c"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.0.4+0"

[[deps.JuliaInterpreter]]
deps = ["CodeTracking", "InteractiveUtils", "Random", "UUIDs"]
git-tree-sha1 = "10da5154188682e5c0726823c2b5125957ec3778"
uuid = "aa1ae85d-cabe-5617-a682-6adf51b2e16a"
version = "0.9.38"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

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
git-tree-sha1 = "854a9c268c43b77b0a27f22d7fab8d33cdb3a731"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+1"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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
git-tree-sha1 = "0b4a5d71f3e5200a7dff793393e09dfc2d874290"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+1"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "6f73d1dd803986947b2c750138528a999a6c7733"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.6.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c6ce1e19f3aec9b59186bdf06cdf3c4fc5f5f3e6"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.50.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "84eef7acd508ee5b3e956a2ae51b05024181dee0"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+0"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "Pkg", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "3eb79b0ca5764d4799c06699573fd8f533259713"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.4.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+0"

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
git-tree-sha1 = "a2d09619db4e765091ee5c6ffe8872849de0feea"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.28"

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
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "1a27764e945a152f7ca7efa04de513d473e9542e"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.14.1"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

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
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "dfdf5519f235516220579f949664f1bf44e741c5"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.6.3"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

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
git-tree-sha1 = "b434dce10c0290ab22cb941a9d72c470f304c71d"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.35.8"

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
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

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
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt5Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Pkg", "Xorg_libXext_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "xkbcommon_jll"]
git-tree-sha1 = "0c03844e2231e12fda4d0086fd7cbe4098ee8dc5"
uuid = "ea2cea3b-5b76-57ae-a6ef-0a8af62496e1"
version = "5.15.3+2"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

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

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

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
deps = ["CodeTracking", "Distributed", "FileWatching", "JuliaInterpreter", "LibGit2", "LoweredCodeUtils", "OrderedCollections", "REPL", "Requires", "UUIDs", "Unicode"]
git-tree-sha1 = "470f48c9c4ea2170fd4d0f8eb5118327aada22f5"
uuid = "295af30f-e4ad-537b-8983-00126c2a3abe"
version = "3.6.4"

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

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "d0553ce4031a081cc42387a9b9c8441b7d99f32d"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.7"

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
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "777657803913ffc7e8cc20f0fd04b634f871af8f"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.8"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

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

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

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
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

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
git-tree-sha1 = "7558e29847e99bc3f04d6569e82d0f5c54460703"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+1"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "93f43ab61b16ddfb2fd3bb13b3ce241cafb0e6c9"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.31.0+0"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "a54ee957f4c86b526460a720dbc882fa5edcbefc"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.41+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "afead5aba5aa507ad5a3bf01f58f82c8d1403495"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+0"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+1"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "12e0eb3bc634fa2080c1c37fccf56f7c22989afd"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.0+4"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+1"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d2d1a5c49fae4ba39983f63de6afcbea47194e85"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+0"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libX11_jll"]
git-tree-sha1 = "0e0dc7431e7a0587559f9294aeec269471c991a4"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "5.0.3+4"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "89b52bc2160aadc84d707093930ef0bffa641246"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.7.10+4"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll"]
git-tree-sha1 = "26be8b1c342929259317d8b9f7b53bf2bb73b123"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.4+4"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "34cea83cb726fb58f325887bf0612c6b3fb17631"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.2+4"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "47e45cd78224c53109495b3e324df0c37bb61fbe"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+0"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+1"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "bcd466676fef0878338c61e655629fa7bbc69d8e"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+0"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "730eeca102434283c50ccf7d1ecdadf521a765a4"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+0"

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
git-tree-sha1 = "330f955bc41bb8f5270a369c473fc4a5a4e4d3cb"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

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
git-tree-sha1 = "b70c870239dc3d7bc094eb2d6be9b73d27bef280"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

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
git-tree-sha1 = "9c304562909ab2bab0262639bd4f444d7bc2be37"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+1"
"""

# ╔═╡ Cell order:
# ╟─2dd6ec04-37fd-11ed-147f-5bc50a90e4bb
# ╟─eeb9e3c7-974d-4f30-b960-f31b952cb79d
# ╟─4e1ddb4c-eea9-4bdf-8183-8addc3c38891
# ╟─92072c1d-9ee7-4d99-826e-146427d0d85d
# ╟─0120c238-e093-41e2-b2e0-53383a1e4880
# ╟─03701ff0-0166-4eed-a326-facd51450cae
# ╟─43a70396-26bc-4d24-b17b-5ec3daaccad5
# ╟─9211c8ac-43b7-43e3-846c-f3764e79706c
# ╟─c74cef62-a137-45e2-960c-dcc0944810fb
# ╟─95b544dd-d3b4-4c64-8269-9a0bf1013b54
# ╟─ca80d078-c859-4cb0-8f5b-ff4ec88848aa
# ╟─d8c17f75-000c-4999-9aed-f2f92d6edeb2
# ╟─166c832c-a4ff-470e-818c-e66ccbba2fac
# ╟─1fae859c-bdfa-41d1-8b8b-f066cb6d9ee9
# ╟─f2172897-1f13-49ed-a5ce-07adf3e61899
# ╟─5a912a9a-1f4d-4841-9070-3c866b43aae2
# ╟─26a21968-819c-4cb1-9c71-16f54599b7de
# ╟─4e0adc00-d644-4290-8b26-c034945992f7
# ╟─a8be829e-81d9-437c-8df1-2673ff3233e4
# ╟─8da3fb38-3ed4-494c-be8b-e8ff92e8bf90
# ╟─6d3a0b93-6d11-4a21-9618-245b8d7e0816
# ╟─160853f2-35fd-4219-9d7a-94baee003888
# ╟─fb9a604e-2a79-4832-8eab-c2f251f9947f
# ╟─aaf0aad5-97c7-4ee8-9081-f078a14a6aec
# ╟─d13c1452-56db-4d3e-9401-59797d5a8ffc
# ╟─5856496b-a0c2-48d6-874e-80b46b0297d7
# ╟─2494ef53-fc58-4152-82f2-8334778eb3c9
# ╟─1d46065d-318e-49d7-b9e2-16fff9222a68
# ╟─c355a185-e35a-429b-a359-4356623a5f22
# ╟─b5a21bae-66cf-4c07-aa67-69987c7cd459
# ╟─5209f762-3eb6-432a-8e4c-a0aa2cde833c
# ╟─962da2f3-4a9b-4cf4-80af-fac1253231af
# ╟─9d61b845-017d-4db6-a51a-c52a85747de5
# ╟─6a4fa5f4-7e21-4528-8792-98a1a6ce8b82
# ╟─9a19c0d8-6ce6-4597-ae10-665aa088b820
# ╟─4e883d98-8f75-4bcf-b2d2-2d06e8595c91
# ╟─252602ae-c223-43da-a3fa-8a59dfc02987
# ╟─e1657fab-1783-4ad6-b292-d7eab019341d
# ╟─af69eea3-8e46-4b22-8d27-c444ac41f090
# ╟─b758d849-44a1-48ca-8c1b-9066965d83a6
# ╟─95dd0d16-00b4-455e-899d-1ab93f035946
# ╟─324d4b06-fe82-472d-9153-cb0665234d9b
# ╟─40d78c6c-bf36-48b8-abef-02721b57d305
# ╟─76d385de-4eb4-4b4b-aaf3-659b6a31691a
# ╟─4f32a716-6be3-4aa6-8745-c586ac80e248
# ╟─b6999f5f-c92a-4a30-bf32-336604490512
# ╟─6d936c43-1c14-4cb9-b7fe-b826bf4f6a8d
# ╟─b2147c2b-7677-4c51-9cd1-1823b09e930e
# ╟─37320389-2c6d-4e18-a51e-6110d646d543
# ╟─a2bdbcea-372a-49d1-b3ac-863b16693afd
# ╟─1d9f5511-2aca-49ce-b413-11a4521a9222
# ╟─7f40c895-ea88-4308-9dad-4ae910082160
# ╟─01c7569e-c635-4c47-a882-4c55cc33f79e
# ╟─7a4bfa6b-f485-4074-b03c-358f6e66102f
# ╟─5dbd5b65-7710-41d4-a7f5-7df1fe4c780c
# ╟─f8549e05-bb5d-43fa-846f-434d0dc5d434
# ╟─6d0a9997-449b-44fd-b596-8d5533fbd77b
# ╟─6ffd8651-e50b-4214-a04f-df5fe8ab1464
# ╟─4cdecd2c-cb2a-4c32-8e94-dd20d878d990
# ╟─9de153f8-6874-4f9d-8fcb-9ccdc5ed9854
# ╟─ecdaabd0-66b6-40c0-b485-25e14e814115
# ╟─451cc71b-6dbd-4cc5-957d-78a4fffd1913
# ╟─8abeb09c-3246-429f-b818-fd6e16d4d862
# ╟─ef37360b-ee56-4727-ace9-8923c1ef98ac
# ╟─5470b98b-0a67-435c-90dd-484c97190663
# ╟─3edbd822-772d-4e49-a96c-d46844ddbf15
# ╟─03c37940-c5d7-4b9d-85bc-2427bb89c5e5
# ╟─41acf75a-50da-41ed-8693-1f7426e48580
# ╟─560cc996-3059-48b4-a674-9b9bcc18f48e
# ╟─40cca076-c520-492b-adc9-4f4cf7c342fc
# ╟─6098f85c-43b4-4298-8ba3-abf390a72cc9
# ╟─8142d3bb-df1f-4865-9e98-fbb435199cf8
# ╟─a392d776-182d-4a84-91c9-923fb16acc6a
# ╟─e6cd70fc-deb9-4698-af35-23c33ca82391
# ╟─e69fb839-218d-47ea-9c45-2236a3c82d0a
# ╟─d22bd0dd-8db4-43bd-b3dc-c4d7ef4c30c3
# ╟─b81367c2-294b-42ef-8fd5-3157aeaf83ee
# ╟─fc661023-20a7-414e-9e14-d01448a0f5b1
# ╟─4c532bc1-124f-4e85-ac5f-6c68a5d33943
# ╟─ee4f5fa0-4260-46fb-9186-15bbdfe34998
# ╟─e4724627-a016-4ae6-83ec-6bd8ca270e30
# ╟─0debcf0e-7680-4183-8332-e28eef0dabb6
# ╟─506c8bd2-6256-407e-b48a-d7fbc73b3a38
# ╟─d24f59ee-c4ed-413b-a8e1-497b31f1fed1
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
