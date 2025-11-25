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

# ╔═╡ c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
begin
	using PlutoTeachingTools
	using PlutoUI
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
	# using PlutoUI

	using Plots; Plots.default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	
end

# ╔═╡ 1c626053-7907-4636-bcac-494f696c4f4b
# begin
# 	using TikzGraphs
# 	using Graphs
# 	using TikzPictures # this is required for saving
# end;

# ╔═╡ 1e916f0b-d8ef-476a-8702-4e7934830eb0
# using Plots;Plots.default(fontfamily="Computer Modern", framestyle=:box)

# ╔═╡ 9b38bfda-c39f-4ac7-8e6c-a264d117eb81
# begin

# 	using PlutoTeachingTools
# 	using PlutoUI
# 	using Random
# 	using LinearAlgebra, LogExpFunctions, Distributions
# end

# ╔═╡ 60c3a104-3761-4079-8704-4642885ab956
# Plots.plot(1:10)

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

# ╔═╡ 56013389-2242-4f23-b272-558fcdecd3b1
# begin
# 	using TikzGraphs
# 	using Graphs
# end

# ╔═╡ 409d6b4d-f0c3-44b0-bfab-7331f434de9b
# begin
# 	g = DiGraph(3)
# 	add_edge!(g, 1, 2)
# 	add_edge!(g, 1, 3)
# 	# add_edge!(g, 1, 3)
# 	graphplt = TikzGraphs.plot(g, [L"\textit{Coin}", L"Y_1", L"Y_2"], options="scale=2, font=\\Huge", node_style="draw", graph_options="nodes={draw,circle}")
# end;

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

# ╔═╡ b865931c-6bd3-405b-b8bb-92e3f4b1646d
html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/coinchoice.png' width = '180' /></center>"

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
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[compat]
Distributions = "~0.25.122"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.10"
LogExpFunctions = "~0.3.29"
Plots = "~1.41.2"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.75"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "5f807f4ec98478322c2b5a77386291f7c8ac4e05"

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
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

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
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

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
git-tree-sha1 = "f305bdb91e1f3fcc687944c97f2ede40585b1bd5"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.19"

    [deps.GR.extensions]
    GRIJuliaExt = "IJulia"

    [deps.GR.weakdeps]
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "de439fbc02b9dc0e639e67d7c5bd5811ff3b6f06"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.19+1"

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

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

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
git-tree-sha1 = "ff69a2b1330bcb730b9ac1ab7dd680176f5896b8"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.1010+0"

[[deps.Measures]]
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

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

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "9b8215b1ee9e78a293f99797cd31375471b2bcae"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.1.3"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

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
git-tree-sha1 = "0662b083e11420952f2e62e17eddae7fc07d5997"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.57.0+0"

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
git-tree-sha1 = "26ca162858917496748aad52bb5d3be4d26a228a"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.4"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "Unzip"]
git-tree-sha1 = "7b990898534ea9797bf9bf21bd086850e5d9f817"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.41.2"

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
git-tree-sha1 = "db8a06ef983af758d285665a0398703eb5bc1d66"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.75"

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

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
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

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "4f96c596b8c8258cc7d3b19797854d368f243ddc"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.4"

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
git-tree-sha1 = "5cb3c5d039f880c0b3075803c8bf45cb95ae1e91"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.51+0"

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
# ╟─c9cfb450-3e8b-11ed-39c2-cd1b7df7ca01
# ╟─1c626053-7907-4636-bcac-494f696c4f4b
# ╟─1e916f0b-d8ef-476a-8702-4e7934830eb0
# ╟─9b38bfda-c39f-4ac7-8e6c-a264d117eb81
# ╟─60c3a104-3761-4079-8704-4642885ab956
# ╟─cf07faa6-e571-4c79-9dd2-f03041b99706
# ╟─3f49be0c-84d6-4bb1-adff-ab7f7ce15d05
# ╟─cb9278ae-fd3f-4a2d-b188-ab463006db38
# ╟─748ade25-bb47-4a67-a624-663fb7a5933f
# ╟─344a3078-fad8-4985-8270-c52b7be034f2
# ╟─56013389-2242-4f23-b272-558fcdecd3b1
# ╟─409d6b4d-f0c3-44b0-bfab-7331f434de9b
# ╟─8bd9e5a8-4534-4c2b-b6e6-5736bd9ec745
# ╟─b44bedea-a0d8-497c-9285-ab5f7aed6d60
# ╟─b865931c-6bd3-405b-b8bb-92e3f4b1646d
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
# ╟─8e1c0744-e813-4f53-af1a-27d27ca58769
# ╟─78060a5a-8201-4443-9a45-5b8390c9e702
# ╟─f448ee19-f495-45c5-957d-45e15ac27f5f
# ╟─5078f341-3190-40c0-8952-2fbbd8167c5b
# ╠═4c72a682-1b1d-4c64-afc3-a61fee9b37d1
# ╠═da9325bd-28e9-4317-aba4-0e2a338b6cff
# ╟─f1c3de03-6f24-4dd7-be37-3550118e3e7e
# ╟─430c7352-8857-41bb-b919-7e330f9b84ff
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
