### A Pluto.jl notebook ###
# v0.20.20

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
	using Distributions
	using SpecialFunctions
	using LinearAlgebra
	# using Dagitty
end

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


* #### Two resellers' problem

"""

# ╔═╡ 166c832c-a4ff-470e-818c-e66ccbba2fac
md"""


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
\Large C \; \Rightarrow\; Y_1, Y_2, Y_3
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

* ##### drawing 100 nodes is tedious 



#### *Plate* notation is invented for this
  * ##### just like a `for` or `while` loop


"""

# ╔═╡ 962da2f3-4a9b-4cf4-80af-fac1253231af
html"""

<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/coinguessdagplate.png" width = "200"/></center>

"""

# ╔═╡ 03c37940-c5d7-4b9d-85bc-2427bb89c5e5
begin
	# using TikzGraphs
	# using Graphs
	# using TikzPictures # this is required for saving
end;

# ╔═╡ 4052bc2a-1cc7-406d-b4fb-6bbc0f3ff2fe
md"""
# Bayes' net and machine learning*

#### Binary classification

* #### input features: ``\mathbf{x} \in \mathbb{R}^m``


* #### output label: ``y^{(i)} \in \{0,1\}``

"""

# ╔═╡ 798303a4-b09b-4a83-8d8e-1e3aeb5e6760
md"""

## Logistic regression as infernce 



#### Which decision boundary is more probable ?


$$\Large P(\text{decision bounary} |\{{y}_1, \ldots, y_n\})$$

"""

# ╔═╡ 97bd4d20-3410-4768-9045-ac897ab74bdb
md"""
## Recap: Logistic function 

```math
\Large
\sigma(z) = \frac{1}{1+e^{-z}}
```

* #### where ``z = \mathbf{w}^\top\mathbf{x}+w_0``

"""

# ╔═╡ c271c913-4a8b-43d5-bab9-8dc70402d5f8
wv_ = [1, 1] * 1;

# ╔═╡ 5dda1c13-96b6-4ee1-a029-d0773d475769
md"##### _E.g._ with ``\mathbf{w}=``$(latexify_md(wv_)), and ``w_0 =0``"

# ╔═╡ 618c3f55-b5e4-4dab-942f-b7c7c23824c0
md"""

## Recap: logistic regression


#### Therefore, the parameter ``\Large \mathbf{w}`` determines 
* ##### a prediction for $y$ given input $\mathbf{x}$

```math
\large p(y=1|\mathbf{x}, \mathbf{w}) = \sigma(\mathbf{w}^\top\mathbf{x})
```

* ##### also a decision boundary: $\large \mathbf{x}\in \mathbb{R}^m\; s.t.\; p(y=1|\mathbf{x}, \mathbf{w}) = 0.5$
"""

# ╔═╡ efa88df0-5447-4ff0-9861-593b8d4ddad5
md"Change ``w_1``: $(@bind ww1_ Slider(-1.5:0.01:1.8; default=0.5)) Change ``w_2``: $(@bind ww2_ Slider(-1.5:0.01:1.8; default=0.5)), change angle $(@bind theta Slider(-180:180; default=58))"

# ╔═╡ 06f0bf3d-0585-4f1d-b290-1bccb5be4db2
md"Add decision boundary $(@bind add_db CheckBox())"

# ╔═╡ ce81fe68-7f9d-4c04-8675-83d178c74a80
ww_ = [ww1_, ww2_]

# ╔═╡ b504f9a0-67a0-4039-b3be-767b63e65b09
md"""

## Bayesian logistic regression as a BN

"""

# ╔═╡ 7bfaa11b-636d-4ed0-98f0-36ffdf3a3c2f
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

# ╔═╡ d497e656-32ea-4704-bc9f-bcf645584720
md"""

## The likelihood function
#### _a.k.a_ cross entropy loss

$$\Large\ell(\mathbf{w}) = p(y|\mathbf{w}, \mathbf{x})= \begin{cases} \sigma& y = 1\\ 1- \sigma& y = 0\end{cases}$$

* ##### where $\sigma = \sigma(\mathbf{w}^\top\mathbf{x})$ is the bias of the Bernoulli


* ##### this is called the **likelihood function of $\mathbf{w}$**


* ##### it is a function of the unknown parameter $\mathbf{w}$
  * ##### here both $\mathbf{x}, y$ are considered given and fixed (*i.e.* not variables)

"""

# ╔═╡ 84cba9ec-6f46-4b45-b604-0efe15408fb8
md"""

## The likelihood function
#### _a.k.a_ cross entropy loss

$$\Large\ell(\mathbf{w}) = p(y|\mathbf{w}, \mathbf{x})= \begin{cases} \sigma& y = 1\\ 1- \sigma& y = 0\end{cases}$$

* ##### where $\sigma = \sigma(\mathbf{w}^\top\mathbf{x})$ is the bias of the Bernoulli


* ##### this is called the **likelihood function of $\mathbf{w}$**


## 

#### The function can be rewritten as a one-liner 


$$\Large\begin{align}\ell(\mathbf{w})&= \begin{cases} \sigma& y = 1\\ 1- \sigma& y = 0\end{cases} \\ 
&= \sigma^{y}(1-\sigma)^{1-y} \end{align}$$




"""

# ╔═╡ e24d9d3e-b642-405c-a52e-227538e63797
md"""

## The likelihood function
#### _a.k.a_ cross entropy loss

$$\Large\ell(\mathbf{w}) = p(y|\mathbf{w}, \mathbf{x})= \begin{cases} \sigma& y = 1\\ 1- \sigma& y = 0\end{cases}$$

* ##### where $\sigma = \sigma(\mathbf{w}^\top\mathbf{x})$ is the bias of the Bernoulli


* ##### this is called the **likelihood function of $\mathbf{w}$**


* ##### it is a function of the unknown parameter $\mathbf{w}$
  * ##### here both $\mathbf{x}, y$ are considered given and fixed (*i.e.* not variables)

## The likelihood function and cross entropy

#### The function can be rewritten as a one-liner 


$$\Large\begin{align}\ell(\mathbf{w})&= \begin{cases} \sigma& y = 1\\ 1- \sigma& y = 0\end{cases} \\ 
&= \sigma^{y}(1-\sigma)^{1-y} \end{align}$$


#### If we take log on both sides, we recover the (negative) cross entropy loss


$$\Large\mathscr{L}(\mathbf{w}) = y \ln \sigma+ (1-y)\ln(1-\sigma)$$



$$\Large \text{cross-entropy}(\mathbf{w}) = -y \ln \sigma - (1-y)\ln(1-\sigma)$$



#### therefore, maximising log-likelihood = minimising cross entropy loss

$$\Large\max_{\mathbf{w}} \mathscr{L}(\mathbf{w}) = \min_\mathbf{w}\text{cross-entropy}(\mathbf{w})$$
"""

# ╔═╡ 62e7f402-81cd-46d8-8f31-12f2a7a13a4c
md"""

## Bayesian logistic regression


#### Now learning becomes an ordinary Bayesian inference problem

* #### which $\mathbf{w}$ is more probable ?


$$\Large p(\mathbf{w}|\{{y}_1, \ldots, y_n\})$$

"""

# ╔═╡ e23376c2-7be5-4366-95d2-5f665999b218
Foldable("Bayes' rule", md"""

$$\Large p(\mathbf{w}|\mathbf{y}) =  \frac{p(\mathbf{w}, \mathbf{y})}{p(\mathbf{y})} = \frac{p(\mathbf{w}, \mathbf{y})}{\int p(\mathbf{w}, \mathbf{y})d\mathbf{w}}$$
""")

# ╔═╡ 437a1ed4-8a0d-409d-9e8d-91d63db55028
md"""

# Amazon seller problem



!!! note "Amazon reseller"
	#### Two resellers sell the same product but with different review counts

	| Seller | A | B |
	|:---:|---|---|
	|# Positive ratings  |799  |8 |
	|# Total ratings  |1000|10|
	#### Which one shall I choose?

## One seller problem
"""

# ╔═╡ cbf527e2-7177-412f-b8c5-0c78a958f782
md"""


#### Let's rephrase the problem a bit

* ##### A seller with *an unknown positive rate* is "tossed" 10 times. 8 out of the 10 experiments are positive. 
\


> ##### What is the **unknown positive rate** ?

"""

# ╔═╡ 8667f083-f2e6-4b1e-b3a8-52a373f37b5c
md"""


## One seller's BN

#### The positive rate: ``\theta \in [0,1]``: 

* ##### the unknown positive rate of the seller; 

#### The number of positive ratings: ``N^+ = 8``: 

* ##### 8 out of 10 tosses is positive; ``N^+`` is the observed data
* ##### whereas ``N=10``, the total tosses, is a fixed constant


"""

# ╔═╡ 52716553-75d1-4bde-8da4-d497a4cf21d4
md"""
## One seller's BN
"""

# ╔═╡ 9460105a-a596-4497-90ee-7efefde76dcb
TwoColumn(md"""
\
\


### The unknown positive rate: $\theta$


\
\
\
\



### ``N^+``: number of positive rates



""", html"""<br><br><br><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/oneseller.png" width = "130"/></center>
""")

# ╔═╡ 00a53997-43bc-4f5d-8b4b-0cb226ccc0d9
md"""

## One seller's BN
"""

# ╔═╡ d10f8297-aee0-43b6-8ec7-fec402df74d4
TwoColumn(md"""


#### CPF ``P(\theta)`` (note that ``\theta\in [0,1]``)

* ##### discretize to``\{0, 0.01, 0.02, \ldots, 1.0\}`` 101 values
* ##### a non-information uniform prior
```math
\Large P(\theta) = \begin{cases} \frac{1}{101} & \theta \in \{0, 0.01, \ldots, 1.0\} \\
0 & \text{otherwise}\end{cases}
```





""", html"""<br><br><br><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/oneseller.png" width = "130"/></center>
""")

# ╔═╡ abaf00e3-ce8c-4780-b2a9-2d526d49f9cd
md"""
## Recap: binomial distribution

#### Note that ``N⁺`` is defined as the total count of successes of ``N`` tosses:

```math
\Large N⁺\triangleq \sum_i Y_i 
```

\

#### We already know the number of successes ``N⁺`` is Binomial distributed (lecture 1), *i.e.*

```math
\large P(N⁺ |\theta, N) = \text{Binom}(N⁺; N, \theta) = \binom{N}{N⁺} \theta^{N⁺} (1-\theta)^{N-N⁺}
```


"""

# ╔═╡ 1d30bf23-79b6-41f9-898b-55186114c145
md"""

## One seller's BN
"""

# ╔═╡ acc11d0c-b61e-4113-81c4-61ec1285611b
TwoColumn(md"""


#### CPF ``P(\theta)``
* ##### a non-information uniform prior
```math
\Large P(\theta) = \begin{cases} \frac{1}{101} & \theta \in \{0, 0.01, \ldots, 1.0\} \\
0 & \text{otherwise}\end{cases}
```
\

#### CPF ``P(N^+|\theta)``:

```math
\Large P(N^+|\theta) =  \text{Binom}(N⁺; N, \theta)
```



""", html"""<br><br><br><center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/oneseller.png" width = "130"/></center>
""")

# ╔═╡ 8eff7d3e-e44b-4673-9efd-29ff774f8f29
function ℓ_binom(N⁺, θ, N; logprob=false)
	# log(binomial(N, N⁺) * (1-θ)^(N-N⁺) * θ^N⁺)
	logL = logabsbinomial(N, N⁺)[1] + xlogy(N-N⁺, 1-θ)  + xlogy(N⁺, θ)
	logprob ? logL : exp(logL)
end;

# ╔═╡ 1c7d43c0-02aa-4ddf-8789-4ba9cd97a1e1
md"""

## Two resellers ?

"""

# ╔═╡ a3b79d61-088b-4ec3-82a8-55c626f544ef
TwoColumn(md"""
\


* #### Two unknowns: ``\theta_A,\theta_B``
\

* #### The counts of "heads" (or positive reviews): 
  * ##### ``\large \mathcal{D} = \{N_{A}^+, N_{B}^+\}``




""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/bayes/twoseller.png" width = "300"/></center>
""")

# ╔═╡ d7a15aff-e928-460f-af85-11269758e186
md"""

#### Based on the BN:

```math
\Large P(\theta_A, \theta_B, N_A^+, N_B^+) = {P(\theta_A)P(\theta_B)} \,{P(N^+_A|\theta_A)P(N^+_B|\theta_B)}
```
"""

# ╔═╡ e4272036-bc02-4603-b937-b2b7b16299fd
md"""

## How about ``\theta_A > \theta_B`` ?


#### Simple, add another node!
\


```math

\Large P\Big ((\theta_A>\theta_B)|\theta_A, \theta_B\Big ) = \begin{cases} 
1 & \theta_A>\theta_B\\
0 & \text{otherwise}
\end{cases}
```

* ##### this is a deterministic node 

"""

# ╔═╡ d00793c9-c2d9-4896-a429-9020b6998ac4
html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/tworesellers_bn.svg" width = "400"/></center>
"""

# ╔═╡ ad288cfc-1d6c-4154-bd3c-84df241d8e1f
md"""
## Based on the BN
\


```math
\Large P(\theta_A, \theta_B, N_A^+, N_B^+) = \underbrace{P(\theta_A)P(\theta_B)}_{\text{prior}} \,\underbrace{P(N^+_A|\theta_A)P(N^+_B|\theta_B)}_{\text{likelihood}}
```



* #### we can apply Bayes' rule to compute the posterior

```math
\Large P \big (\theta_A, \theta_B\big | N_A^+, N_B^+\big  ) \propto P(\theta_A, \theta_B, N_A^+, N_B^+) 
```

"""

# ╔═╡ f6d8b85c-af90-443b-9c69-55bdc1711f74
md"""



## Visualisation: prior ``P(\theta_A, \theta_B)``


$$\large P(\theta_A, \theta_B) = P(\theta_A) P(\theta_B)=\begin{cases} 1/101^2, & \theta_A,\theta_B \in \{0, 0.01, \ldots, 1.0\}^2 \\
0, & \text{otherwise}; \end{cases}$$



"""

# ╔═╡ 0c7bd0e2-324b-4da6-8087-1548ad77e848
let	
	gr()
	dis_size = 11
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
 	plt = plot(xlim=[0,1], ylim=[0, 1], yticks=θ₂s, xticks=θ₁s, zlim = [0, 0.1], zaxis=false, framestyle=:origin, camera=(15,15), size=(800, 600), xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="", title="Prior "*L"P(\theta_A,\theta_B)")
	for (i, θ₁) in enumerate(θ₁s)
		for (j, θ₂) in enumerate(θ₁s)
			quiver!([θ₂], [θ₁], [0], quiver=([0], [0], [1/dis_size^2]), lw=3, lc=:orange, alpha=0.9)
		end
	end
	plt
end

# ╔═╡ 0281ea87-3409-43d1-bac1-4f3f14f14690
begin
	dis_size = 101
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
end;

# ╔═╡ 19e5eca4-acb1-4926-bb21-cb712c17586f
md"""
## Visualisation: likelihood ``p(\mathcal{D}|\theta_A, \theta_B)``



```math
\large
\begin{align}
P(\mathcal{D}|\theta_A, \theta_B)
&= \text{Binom}(N_A^{+}; N_A, \theta_A)\times  \text{Binom}(N_B^{+}; N_B, \theta_B) \\
&=\binom{N_A}{N_A^+}\theta_A^{N_A^+}(1-\theta_A)^{N_A- N_A^+}\times \binom{N_B}{N_B^+}\theta_B^{N_B^+}(1-\theta_B)^{N_B- N_B^+}
\end{align}
```
* ##### where ``\mathcal{D} =\{N_A^{+}, N_B^{+}\}``
* ##### and also note that ``N_A=10; N_B=100`` are fixed constants

"""

# ╔═╡ 3e7b2a60-2a18-4a27-a8a8-a53835a75015
md"""

## Visualisation: posterior
* #### apply Bayes' rule

$$\Large \begin{align}
P(\theta_A, \theta_B|\mathcal D) &= \frac{P(\theta_A, \theta_B) P(\mathcal{D}|\theta_A, \theta_B)}{P(\mathcal{D})}\\
&\propto P(\theta_A, \theta_B) P(\mathcal{D}|\theta_A, \theta_B)
\end{align}$$

\

"""

# ╔═╡ e32bbcaf-7a8d-440e-927b-fcac71da6c4e
begin
	N₁, N₂ = 1000, 10
	Nₕ₁, Nₕ₂ = 799, 8
end;

# ╔═╡ 36d2fda2-a950-4c87-8ce1-2303a6d410fc
begin
	function ℓ_two_coins(θ₁, θ₂; N₁=10, N₂=100, Nh₁=7, Nh₂=69, logLik=false)
		logℓ = ℓ_binom(Nh₁, θ₁, N₁;  logprob=true) + ℓ_binom(Nh₂, θ₂, N₂; logprob=true)
		logLik ? logℓ : exp(logℓ)
	end

	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=true) for θᵢ in θ₁s, θⱼ in θ₂s]
	p𝒟 = exp(logsumexp(ℓ_twos))
	ps = exp.(ℓ_twos .- logsumexp(ℓ_twos))
end;

# ╔═╡ 93ab13ae-1aea-4eb9-81c5-ac7800ffa3fc
let	
	gr()
	dis_size = 21
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=false) for θᵢ in θ₁s, θⱼ in θ₂s]
	ℓ_twos_vc = ℓ_twos ./maximum(ℓ_twos)
	cv=cgrad(:jet, rev = false)
 	plt = plot(xlim=[0,1], ylim=[0, 1], yticks=θ₂s[1:2:end], xticks=θ₁s[1:2:end],  zaxis=false, framestyle=:origin, camera=(30,15), size=(800, 600), zticks =false, xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel="", title="Likelihood: " *L"P(\mathcal{D}|\theta_A,\theta_B)")
	for (i, θ₁) in enumerate(θ₁s)
		for (j, θ₂) in enumerate(θ₁s)
			quiver!([θ₂], [θ₁], [0], quiver=([0], [0], [ℓ_twos_vc[j,i]]), lw=3,lc=cv[ℓ_twos_vc[j,i]],  alpha=0.9)
		end
	end
	plt
end

# ╔═╡ 89276fce-1f90-459b-93bf-5a223edf21e0
let	
	gr()
	dis_size = 21
	θ₁s, θ₂s = range(0, 1 , dis_size), range(0, 1 , dis_size)
	ℓ_twos = [ℓ_two_coins(θᵢ, θⱼ; N₁=N₁, N₂=N₂, Nh₁=Nₕ₁, Nh₂=Nₕ₂, logLik=false) for θᵢ in θ₁s, θⱼ in θ₂s]
	ℓ_twos_vc = ℓ_twos ./sum(ℓ_twos)
	ℓ_twos_vc_ = ℓ_twos ./maximum(ℓ_twos)
	cv = cgrad(:jet, rev = false)
 	plt = plot(xlim=[0,1], ylim=[0, 1], yticks=θ₂s[1:2:end], xticks=θ₁s[1:2:end],   framestyle=:origin, camera=(30,15), size=(800, 600), xlabel=L"\theta_A", ylabel=L"\theta_B", zlabel=L"P(\theta_A, \theta_B|\mathcal{D})", title="Posterior: " *L"P(\theta_A,\theta_B | \mathcal{D})")
	for (i, θ₁) in enumerate(θ₁s)
		for (j, θ₂) in enumerate(θ₁s)
			quiver!([θ₂], [θ₁], [0], quiver=([0], [0], [ℓ_twos_vc[j,i]]), lw=3,lc=cv[ℓ_twos_vc_[j,i]],  alpha=0.9)
		end
	end
	plt
end

# ╔═╡ c260decc-e8a9-41c6-9e5e-a5c2204db27d
md"""

## Visualisation -- surface plot



"""

# ╔═╡ 22387190-dcc5-441a-8056-419762976aa4
md"""
* #### note how narrow ``\theta_A``'s distribution is, comparing with ``\theta_B``

"""

# ╔═╡ 29c57b42-17c7-47ce-aab2-3bc7d2844a04
begin
	plotly()
	p2 = Plots.plot(θ₁s, θ₂s,  ps', st=:surface, xlabel= "θA", ylabel="θB", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.0005, maximum(ps)], colorbar=false, c=:jet, zlabel="", alpha=0.9, title="Posterior")
end

# ╔═╡ c5899523-b405-4a15-be79-72c81742d20c
md"""


## Visualisation -- heat map




$(begin
gr()
Plots.plot(θ₁s, θ₂s,  ps', st=:heatmap, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], zlim =[-0.003, maximum(ps)], colorbar=false, c=:plasma, zlabel="density", alpha=0.7, title="Posterior's density heapmap")

end)
"""

# ╔═╡ a7c6daa1-7cda-469f-8350-43e4edcc184d
md"""

## Lastly: answer the question

#### The question is

```math
\Large P(\theta_A > \theta_B|\mathcal{D})
```
* ##### *In light of the data, how likely coin ``A`` has a higher bias than coin ``B``?*


* ##### which can be computed as 

```math
\large P(\theta_A > \theta_B|\mathcal{D}) = \sum_{\theta_A > \theta_B:\;\theta_A,\theta_B\in \{0.0, \ldots 1.0\}^2} P(\theta_A, \theta_B|\mathcal{D})
```
  * ##### geometrically, calculate the probability below the shaded area


"""

# ╔═╡ fa4b41fd-0195-410b-b3ce-89a4602ea7ee
md"""

#### For our problem, as expected
* ##### 61% chance seller A is better!
* ##### it is only about 39% chance that seller B is better

"""

# ╔═╡ 5d127bfc-81c6-4ec3-a7c1-32e21a3ada57
post_AmoreB = let
	post_AmoreB = 0.0
	for (i, θᵢ) in enumerate(θ₁s)
		for (j, θⱼ) in enumerate(θ₂s)
			if θᵢ > 1.0 * θⱼ
				post_AmoreB += ps[i,j]
			end
		end
	end
	post_AmoreB
end;

# ╔═╡ 167406e7-2213-4bc8-a227-441077fe9e81
begin
	gr()
	post_p_amazon =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin)
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright, title=L"p(\theta_A > \theta_B|\mathcal{D}) \approx %$(round(post_AmoreB, digits=2))")
end

# ╔═╡ 4ff958a4-4449-465c-be8e-289040710ee5
md"""

## Question


!!! question "Question"
	##### What's the chance that seller A's positive rate is **twice better** than seller B?

"""

# ╔═╡ c2fdb611-d8b8-4298-8b6a-f774fbd68248
md"""

!!! hint "Answer"

	Just introduce ``k=2`` below

	```math
	P(\theta_A > k \cdot \theta_B|\mathcal{D}) = \sum_{\theta_A \,>\, k\times  \theta_B} p(\theta_A, \theta_B|\mathcal{D})
	```
	
	* a different shaded area!


"""

# ╔═╡ 6e6d936a-70e7-4d0d-bda6-7fd412af5acd
md"Move me: $(@bind k Slider(0.1:0.1:3; default =1))"

# ╔═╡ 1b1076e1-4ab0-4266-867c-df4181f3c4e5
post_AmoreKB = let
	post_AmoreKB = 0
	for (i, θᵢ) in enumerate(θ₁s)
		for (j, θⱼ) in enumerate(θ₂s)
			if θᵢ > k*θⱼ
				post_AmoreKB += ps[i,j]
			end
		end
	end
	post_AmoreKB
end;

# ╔═╡ 5e32c50e-5802-4826-bd38-92c5d52e5708
let
	gr()
	post_p_amazon =Plots.plot(θ₁s, θ₂s,  ps', st=:contour, xlabel=L"\theta_A", ylabel=L"\theta_B", ratio=1, xlim=[0,1], ylim=[0,1], colorbar=false, c=:thermal, framestyle=:origin)
	plot!((x) -> x, lw=1, lc=:gray, label="")
	equalline = Shape([(0., 0.0), (1,1), (1, 0)])
	plot!(equalline, fillcolor = plot_color(:gray, 0.2), label=L"\theta_A>\theta_B", legend=:bottomright)
	kline = Shape([(0., 0.0), (1,1/k), (1, 0)])
	plot!(post_p_amazon, kline, fillcolor = plot_color(:gray, 0.5), label=L"\theta_A>%$(k)\cdot \theta_B", legend=:bottomright, title =L"P(\theta_A > %$(k)\, \theta_B|\mathcal{D})\approx %$(round(post_AmoreKB, digits=2))")
end

# ╔═╡ 735f5b42-99fd-41cd-9259-2ece99f976ac
md"""
## Appendix

"""

# ╔═╡ c2d268f7-7b92-40ef-a700-678750c80080
# only works for uni-modal
function find_hpdi(ps, α = 0.95)
	cum_p, idx = findmax(ps)
	l = idx - 1
	u = idx + 1
	while cum_p <= α
		if l >= 1 
			if u > length(ps) || ps[l] > ps[u]
				cum_p += ps[l]
				l = max(l - 1, 0) 
				continue
			end
		end
		
		if u <= length(ps) 
			if l == 0 || ps[l] < ps[u]
				cum_p += ps[u]
				u = min(u + 1, length(ps))
			end
		end
	end
	return l+1, u-1, cum_p
end;

# ╔═╡ 295da8be-bbd4-4437-a7b6-48b8b3e0bc60
md"""
#### Data generation
"""

# ╔═╡ a729e2fe-c27a-4cfd-b126-c08c36a4d5d9
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

# ╔═╡ e7438b3e-9815-4954-86e1-020222a78a12
let
	gr()
	plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
end

# ╔═╡ 98500504-f082-40f9-8774-8a2e46607469
let
	gr()
	plt = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2, ratio=1)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
	Random.seed!(123)

	n = 8
	ws = randn(n, 2) ./ 2

	ws[:, 2] .= randn(n) .- 1

	for (i, w) in enumerate(eachrow(ws))
		plot!(-7:0.5:7, x-> w[1] + w[2]*x, lc=:gray, label= i == 1 ? "Decision boundaries" : "")
		
	end

	plt
end

# ╔═╡ 9a8844f2-a649-4aaf-87c8-649f6c8af2df
let
	gr()
	plt = scatter(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], ones(sum(targets_D₂ .== 1)),  label="y=1", c=2)
	scatter!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], 0 *ones(sum(targets_D₂ .== 0)), label="y=0", framestyle=:zerolines, c=1)
	# w = linear_reg(D₂, targets_D₂;λ=0.0)
	f̂(x, y) = logistic(0 + ww_[1] * x + ww_[2] * y)
	plot!(-8:.5:8, -8:0.5:8, (x,y) -> f̂(x, y), alpha =0.4, st=:surface, colorbar=false,c=:jet, title="Logistic regression", camera=(theta,20), xlabel=L"x_1", ylabel=L"x_2", zlabel=L"y")

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

# ╔═╡ 44ca90d5-47ff-47ec-a0af-46282c47afef
let
	gr()
	plt = plot(D₂[targets_D₂ .== 1, 2], D₂[targets_D₂ .== 1, 3], st=:scatter, label=L"y^{(i)} = 1", xlabel=L"x_1", ylabel=L"x_2", c=2, ratio=1)
	plot!(D₂[targets_D₂ .== 0, 2], D₂[targets_D₂ .== 0, 3], st=:scatter, c=1, framestyle=:origin, label=L"y^{(i)} = 0", xlim=[-8, 8], ylim=[-6, 6])
	Random.seed!(123)

	n = 8
	ws = randn(n, 2) ./ 2

	ws[:, 2] .= randn(n) .- 1

	for (i, w) in enumerate(eachrow(ws))
		plot!(-7:0.5:7, x-> w[1] + w[2]*x, lc=:gray, label= i == 1 ? L"\mathbf{w}"*"'s decision boundary" : "")
		
	end

	plt
end

# ╔═╡ 27a5b8a7-a62d-4be5-aa38-87eac18ca883
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

# ╔═╡ ededbf8b-672c-485d-a938-521c82a243ee
begin
	n3_ = 30
	extraD = randn(n3_, 2)/2 .+ [2 -6]
	D₃ = [copy(D₂); [ones(n3_) extraD]]
	targets_D₃ = [targets_D₂; zeros(n3_)]
	targets_D₃_ = [targets_D₂; -ones(n3_)]
end

# ╔═╡ fc72d5ce-fb2c-413d-9434-cfc93e771775
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

# ╔═╡ 49b5b9b9-a8c7-411d-ad02-e8fa85b07b4f
let
	plotly()
	w₀ = 0
	w₁, w₂ = wv_[1], wv_[2]
	p1 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> w₀+ w₁* x + w₂ * y, st=:surface, c=:jet, colorbar=false, alpha=0.8, framestyle=:zerolines, xlabel="x₁", ylabel="x₂", title="z(x) = wᵀ x")

	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0, st=:surface, c=:gray, alpha=0.5)
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	x0s = -5:0.5:5
	if w₂ ==0
		x0s = range(-w₀/w₁-eps(1.0) , -w₀/w₁+eps(1.0), 20)
		y0s = range(-5, 5, 20)
	else
		y0s = (- w₁ * x0s .- w₀) ./ w₂
	end
	plot!(x0s, y0s, zeros(length(x0s)), lc=:gray, lw=4, label="")
	
	p2 = plot(-5:0.5:5, -5:0.5:5, (x, y) -> logistic(w₀+ w₁* x + w₂ * y), st=:surface, c=:jet, colorbar=false, alpha=0.8, zlim=[-0.1, 1.1],  xlabel="x₁", ylabel="x₂", title="σ(wᵀx)", framestyle=:zerolines)
	plot!(-5:0.5:5, -5:0.5:5, (x, y) -> 0.5, st=:surface, c=:gray, alpha=0.75)
	arrow3d!([0], [0], [0], [w₁], [w₂], [0]; as=0.5, lc=2, la=1, lw=2, scale=:identity)
	plot!(x0s, y0s, .5 * ones(length(x0s)), lc=:gray, lw=4, label="")
	plot(p1, p2)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "33083766b0daa603c567d6a2b772e521689a8f44"

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
git-tree-sha1 = "7e35fca2bdfba44d797c53dfe63a51fabf39bfc0"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.4.0"
weakdeps = ["SparseArrays", "StaticArrays"]

    [deps.Adapt.extensions]
    AdaptSparseArraysExt = "SparseArrays"
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "3e22db924e2945282e70c33b75d4dde8bfa44c94"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.8"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "b0fd3f56fa442f81e0a47815c92245acfaaa4e34"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.31.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "8b3b6f87ce8f65a2b4f857528fd8d70086cd72b1"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.11.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "37ea44092930b1811e666c3bc38065d7d87fcc74"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.1"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "9d8a54ce4b17aa5bdce0ea5c34bc5e7c340d16ad"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.1"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

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
deps = ["OrderedCollections"]
git-tree-sha1 = "6c72198e6a101cccdd4c9731d3985e904ba26037"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.1"

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
git-tree-sha1 = "473e9afc9cf30814eb67ffa5f2db7df82c3ad9fd"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.16.2+0"

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
git-tree-sha1 = "27af30de8b5445644e8ffe3bcb0d72049c089cf1"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.7.3+0"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "95ecf07c2eea562b5adbd0696af6db62c0f52560"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.5"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "ccc81ba5e42497f4e76553a5545665eed577a663"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "8.0.0+0"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "Libdl", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "97f08406df914023af55ade2f843c39e99c5d969"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.10.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6d6219a004b8cf1e0b4dbe27a2860b8e04eba0be"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.11+0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "173e4d8f14230a7523ae11b9a3fa9edb3e0efd78"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.14.0"
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

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7a214fdac5ed5f59a22c2d9a885a16da1c74bbc7"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.17+0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "f52c27dd921390146624f3aab95f4e8614ad6531"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.18"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "4b0406b866ea9fdbaf1148bc9c0b887e59f9af68"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.18+0"

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

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "5e6fe50ae7f23d171f44e311c2960294aaa0beb5"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.19"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "f923f9a774fcf3f5cb761bfa43aeadd689714813"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.1+0"

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

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "ec1debd61c300961f98064cfb21287613ad7f303"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2025.2.0+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "65d505fa4c0d7072990d659ef3fc086eb6da8208"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.16.2"

    [deps.Interpolations.extensions]
    InterpolationsForwardDiffExt = "ForwardDiff"
    InterpolationsUnitfulExt = "Unitful"

    [deps.Interpolations.weakdeps]
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["REPL", "Random", "fzf_jll"]
git-tree-sha1 = "82f7acdc599b65e0f8ccd270ffa1467c21cb647b"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.11"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "ba51324b894edaf1df3ab16e2cc6bc3280a2f1a7"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.10"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "059aabebaa7c82ccb853dd4a0ee9d17796f7e1bc"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.3+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "aaafe88dccbd957a8d82f7d05be9b69172e0cee3"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.1+0"

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

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

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

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "d36c21b9e7c172a44a10484125024495e2625ac0"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.1+1"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "f04133fe05eff1667d2054c53d59f9122383fe05"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.2+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2a7a12fc0a4e7fb773450d17975322aa77142106"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.41.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

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
git-tree-sha1 = "f00544d95982ea270145636c181ceda21c4e2575"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.2.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "282cadc186e7b2ae0eeadbd7a4dffed4196ae2aa"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2025.2.0+0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3cce3511ca2c6f87b19c34ffc623417ed2798cbd"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.10+0"

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
version = "2025.5.20"

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "ca7e18198a166a1f3eb92a3650d53d94ed8ca8a1"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.22"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "117432e406b5c023f665fa73dc26e79ec3630151"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.17.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6aa4566bb7ae78498a5e68943863fa8b5231b59"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.6+0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "NetworkOptions", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "386b47442468acfb1add94bf2d85365dea10cbab"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.6.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1346c9208249809840c91b26703912dff463d335"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.6+0"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c392fc5dd032381919e3b22dd32d6443760ce7ea"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.5.2+0"

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

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "12ce661880f8e309569074a61d3767e5756a199f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.1"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "3faff84e6f97a7f18e0dd24373daa229fd358db5"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.73"

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

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "34f7e5d2861083ec7596af8b8c092531facf2192"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.8.2+2"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "da7adf145cce0d44e892626e647f9dcbe9cb3e10"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.8.2+1"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "9eca9fc3fe515d619ce004c83c31ffd3f85c7ccf"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.8.2+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "8f528b0851b5b7025032818eb5abbeb8a736f853"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.8.2+2"

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
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
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

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

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
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "95af145932c2ed859b63329952ce8d633719f091"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.3"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "b8693004b385c842357406e3af647701fe783f98"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.15"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

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
git-tree-sha1 = "a136f98cefaf3e2924a66bd75173d1c891ab7453"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.7"

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

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "88cf3587711d9ad0a55722d339a013c4c56c5bbc"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.8"

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
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

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

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

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

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "96478df35bbc2f3e1e791bc7a3d0eeee559e60e9"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.24.0+0"

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

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee71455b0aaa3440dfdd54a9a36ccef829be7d4"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.8.1+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a3ea76ee3f4facd7a64684f9af25310825ee3668"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.2+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "9c7ad99c629a44f81e7799eb05ec2746abb5d588"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.6+0"

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

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "6c74ca84bbabc18c4547014765d194ff0b4dc9da"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.4+0"

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

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "75e00946e43621e09d431d9b95818ee751e6b2ef"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.2+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "a376af5c7ae60d29825164db40787f15c80c7c54"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.3+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a5bc75478d323358a90dc36766f3c99ba7feb024"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.6+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "aff463c82a773cb86061bce8d53a0d976854923e"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.5+0"

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

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "e3150c7400c41e207012b41659591f083f3ef795"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.3+0"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "9750dc53819eba4e9a20be42349a6d3b86c7cdf8"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.6+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f4fc02e384b74418679983a97385644b67e1263b"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll"]
git-tree-sha1 = "68da27247e7d8d8dafd1fcf0c3654ad6506f5f97"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "44ec54b0e2acd408b0fb361e1e9244c60c9c3dd4"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.1+0"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "5b0263b6d080716a02544c55fdff2c8d7f9a16a0"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.10+0"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_jll"]
git-tree-sha1 = "f233c83cad1fa0e70b7771e0e21b061a116f2763"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.2+0"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "801a858fc9fb90c11ffddee1801bb06a738bda9b"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.7+0"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "00af7ebdc563c9217ecc67776d1bbf037dbcebf4"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.44.0+0"

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

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "c3b0e6196d50eab0c5ed34021aaa0bb463489510"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.14+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b6a34e0e0960190ac2a4363a1bd003504772d631"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.61.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "371cc681c00a3ccc3fbc5c0fb91f58ba9bec1ecf"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.13.1+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "125eedcb0a4a0bba65b657251ce1d27c8714e9d6"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.17.4+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "56d643b57b188d30cccc25e331d416d3d358e557"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.13.4+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "646634dd19587a56ee2f1199563ec056c5f228df"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.4+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "91d05d7f4a9f67205bd6cf395e488009fe85b499"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.28.1+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "07b6a107d926093898e82b3b1db657ebe33134ec"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.50+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b4d631fd51f2e9cdd93724ae25b2efc198b059b1"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.7+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "1350188a69a6e46f799d3945beef36435ed7262f"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2022.0.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "14cc7083fc6dff3cc44f2bc435ee96d06ed79aa7"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "10164.0.1+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e7b67590c14d487e734dcb925924c5dc43ec85f3"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "4.1.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "fbf139bce07a534df0e699dbb5f5cc9346f95cc1"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.9.2+0"
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
# ╟─03c37940-c5d7-4b9d-85bc-2427bb89c5e5
# ╟─4052bc2a-1cc7-406d-b4fb-6bbc0f3ff2fe
# ╟─e7438b3e-9815-4954-86e1-020222a78a12
# ╟─798303a4-b09b-4a83-8d8e-1e3aeb5e6760
# ╟─98500504-f082-40f9-8774-8a2e46607469
# ╟─97bd4d20-3410-4768-9045-ac897ab74bdb
# ╟─5dda1c13-96b6-4ee1-a029-d0773d475769
# ╟─c271c913-4a8b-43d5-bab9-8dc70402d5f8
# ╟─49b5b9b9-a8c7-411d-ad02-e8fa85b07b4f
# ╟─618c3f55-b5e4-4dab-942f-b7c7c23824c0
# ╟─efa88df0-5447-4ff0-9861-593b8d4ddad5
# ╟─06f0bf3d-0585-4f1d-b290-1bccb5be4db2
# ╟─ce81fe68-7f9d-4c04-8675-83d178c74a80
# ╟─9a8844f2-a649-4aaf-87c8-649f6c8af2df
# ╟─b504f9a0-67a0-4039-b3be-767b63e65b09
# ╟─7bfaa11b-636d-4ed0-98f0-36ffdf3a3c2f
# ╟─d497e656-32ea-4704-bc9f-bcf645584720
# ╟─84cba9ec-6f46-4b45-b604-0efe15408fb8
# ╟─e24d9d3e-b642-405c-a52e-227538e63797
# ╟─62e7f402-81cd-46d8-8f31-12f2a7a13a4c
# ╟─e23376c2-7be5-4366-95d2-5f665999b218
# ╟─44ca90d5-47ff-47ec-a0af-46282c47afef
# ╟─437a1ed4-8a0d-409d-9e8d-91d63db55028
# ╟─cbf527e2-7177-412f-b8c5-0c78a958f782
# ╟─8667f083-f2e6-4b1e-b3a8-52a373f37b5c
# ╟─52716553-75d1-4bde-8da4-d497a4cf21d4
# ╟─9460105a-a596-4497-90ee-7efefde76dcb
# ╟─00a53997-43bc-4f5d-8b4b-0cb226ccc0d9
# ╟─d10f8297-aee0-43b6-8ec7-fec402df74d4
# ╟─abaf00e3-ce8c-4780-b2a9-2d526d49f9cd
# ╟─1d30bf23-79b6-41f9-898b-55186114c145
# ╟─acc11d0c-b61e-4113-81c4-61ec1285611b
# ╟─8eff7d3e-e44b-4673-9efd-29ff774f8f29
# ╟─36d2fda2-a950-4c87-8ce1-2303a6d410fc
# ╟─1c7d43c0-02aa-4ddf-8789-4ba9cd97a1e1
# ╟─a3b79d61-088b-4ec3-82a8-55c626f544ef
# ╟─d7a15aff-e928-460f-af85-11269758e186
# ╟─e4272036-bc02-4603-b937-b2b7b16299fd
# ╟─d00793c9-c2d9-4896-a429-9020b6998ac4
# ╟─ad288cfc-1d6c-4154-bd3c-84df241d8e1f
# ╟─f6d8b85c-af90-443b-9c69-55bdc1711f74
# ╟─0c7bd0e2-324b-4da6-8087-1548ad77e848
# ╟─0281ea87-3409-43d1-bac1-4f3f14f14690
# ╟─19e5eca4-acb1-4926-bb21-cb712c17586f
# ╟─93ab13ae-1aea-4eb9-81c5-ac7800ffa3fc
# ╟─3e7b2a60-2a18-4a27-a8a8-a53835a75015
# ╟─89276fce-1f90-459b-93bf-5a223edf21e0
# ╟─e32bbcaf-7a8d-440e-927b-fcac71da6c4e
# ╟─c260decc-e8a9-41c6-9e5e-a5c2204db27d
# ╟─22387190-dcc5-441a-8056-419762976aa4
# ╟─29c57b42-17c7-47ce-aab2-3bc7d2844a04
# ╟─c5899523-b405-4a15-be79-72c81742d20c
# ╟─a7c6daa1-7cda-469f-8350-43e4edcc184d
# ╟─167406e7-2213-4bc8-a227-441077fe9e81
# ╟─fa4b41fd-0195-410b-b3ce-89a4602ea7ee
# ╟─5d127bfc-81c6-4ec3-a7c1-32e21a3ada57
# ╟─4ff958a4-4449-465c-be8e-289040710ee5
# ╟─c2fdb611-d8b8-4298-8b6a-f774fbd68248
# ╟─6e6d936a-70e7-4d0d-bda6-7fd412af5acd
# ╟─5e32c50e-5802-4826-bd38-92c5d52e5708
# ╟─1b1076e1-4ab0-4266-867c-df4181f3c4e5
# ╟─735f5b42-99fd-41cd-9259-2ece99f976ac
# ╠═c2d268f7-7b92-40ef-a700-678750c80080
# ╟─295da8be-bbd4-4437-a7b6-48b8b3e0bc60
# ╟─a729e2fe-c27a-4cfd-b126-c08c36a4d5d9
# ╟─27a5b8a7-a62d-4be5-aa38-87eac18ca883
# ╟─ededbf8b-672c-485d-a938-521c82a243ee
# ╠═fc72d5ce-fb2c-413d-9434-cfc93e771775
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
