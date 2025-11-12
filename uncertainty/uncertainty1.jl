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

# ╔═╡ 9f90a18b-114f-4039-9aaf-f52c77205a49
begin
	using LinearAlgebra
	using PlutoUI
	using PlutoTeachingTools
	using LaTeXStrings
	using Latexify
	using Random
	using Statistics
	using HypertextLiteral
	using Plots; default(fontfamily="Computer Modern", framestyle=:box) # LaTex-style
	using StatsPlots
	using LogExpFunctions
	using GLM
end

# ╔═╡ 1afdb42f-6bce-4fb1-860a-820d98df0f9d
using Distributions

# ╔═╡ 50752620-a604-442c-bf92-992963b1dd7a
using Images

# ╔═╡ ef112987-74b4-41fc-842f-ebf1c901b59b
using StatsBase

# ╔═╡ 3e2e1ea8-3a7d-462f-ac38-43a087907a14
TableOfContents()

# ╔═╡ e31e77ed-5ed6-4689-bfdc-1de73f6fc7b8
figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/figs4CS5010/";

# ╔═╡ d61aca4b-a8a7-4876-9310-647c34d3f715
bayes_figure_url = "https://leo.host.cs.st-andrews.ac.uk/figs/bayes/";

# ╔═╡ bef9bfb4-0521-4cdb-a638-b16c13a60f10
fig_url = "https://leo.host.cs.st-andrews.ac.uk/figs/";

# ╔═╡ 7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
ChooseDisplayMode()

# ╔═╡ bc96a33d-9011-41ec-a19e-d472cbaafb70
md"""

# CS3105 Artificial Intelligence


### Uncertainty 1 
##### Probability theory
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 2495aa78-b0dd-4245-b79e-2a6ebf1b9e96
md"""

## Intelligence and uncertainty

### Reasoning under uncertainty 

* #### one form of human intelligence


* #### taken for granted, almost like instinct
  * sometimes work well, sometimes fail miserably


"""

# ╔═╡ c12493fd-862d-4391-b76a-62fdc1948f9d

md"""


##### (Some) _sources of uncertainty_: 

* **future** is always uncertain


* **_Prediction_** is uncertain


* data is **_Noisy_** 


* _**Model**_ is uncertain (c.f. Machine Learning)



* _**Changing**_ environment


* and more ...

"""

# ╔═╡ 1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
# md"""

# ## Why probability ?



# !!! note ""
# 	##### ``\;\;\;\;\;\;\;\;\;\;`` _**Uncertainty**_ is everywhere in life and AI

# \
# \

# !!! important ""
# 	##### ``\;\;\;\;\;\;\;\;\;\;`` _Probability theory_ is the tool to deal with **_Uncertainty_** 



# """

# ╔═╡ ff058cce-d46b-4fa1-83a3-9ed7188f12ae

# md"""

# ## Uncertainty is everywhere


# ##### (some) _Sources of uncertainty_: 


# * Data is **_Noisy_** 


# * _**Model**_ is uncertain


# * **_Prediction_** is uncertain


# * _**Changing**_ environment


# * and more ...
# """

# ╔═╡ d15e1205-647e-471a-9d71-24e39a7d4801
md"""
## Some motivating examples  (Amazon reseller)


"""

# ╔═╡ b0ec14d9-0ea0-4c06-a17f-f8bf81c5a1e0
md"""



!!! note "Amazon reseller"
	#### Which reseller to choose?
	* ##### both have 93% + positive ratings
	* ##### but with different number of ratings



"""

# ╔═╡ 3bc38a43-f67b-49c9-a5c9-c7d686e0d675
Resource(figure_url * "amazonreviews.png", :width=>700, :align=>"left")

# ╔═╡ cb80542d-1ad1-402c-9422-18325c554eba
# md"""




# !!! note "Amazon reseller"
# 	Two resellers sell the same product on Amazon but with different review counts. 

# 	| Seller | A | B |
# 	|:---:|---|---|
# 	|#Positive ratings  |8  |799 |
# 	|#Total ratings  |10|1000|
# 	Which one shall I choose?
# """

# ╔═╡ 9a0e74cb-2f4f-42cb-b7bf-2fed62101851
md"""
##

#### It goes without saying: the first reseller with more ratings

* ##### but why more ratings is better?
* ##### how to show/justify the reasoning formally ?

"""

# ╔═╡ 0e1c65f9-e172-4948-a40e-9f681271333e
md"""

## Some motivating examples (coughing example)



!!! note "Cold or Cancer"
	#### Someone in our class has coughed. What is likely the cause ?
	* ##### *cancer* or 
	* ##### *cold* 

\


#### It will be absurd if you think you get cancer for one cough

* ##### but why cold makes better sense? 
* ##### how to show/justify the reasoning formally ?

"""

# ╔═╡ f0bc4e74-ce90-421c-aab2-2c004712e2bd
md"""

## Some motivating examples - (ML - regressions)



#### *Regression problem*: find the relationship between input $X$ and $Y$ (when $Y$ is continuous)


"""

# ╔═╡ 6cc906ec-fb88-4151-9d33-947b5d1510d1
TwoColumnWideRight(
md"""
\
\
\

#### For example, *linear regression*

```math
\Large f(X) = \beta_0 + \beta_1 X 
```
"""
	, 
	html"""<center><img src="https://otexts.com/fpp3/fpp_files/figure-html/SLRpop1-1.png" width = "700"/></center>
"""

	
)

# ╔═╡ 0a9985f8-07e4-4ec0-a9eb-3005f95b7e65
md"""
##


#### Now consider a (non-linear) dataset:
* ##### a linear fit is not good
* ##### pay attention to the gaps in the middle
"""

# ╔═╡ aa271ebf-f9ef-453c-8182-0bba6c4a1e45
md"""
#### Intuitively, the left one looks more reasonable
* ##### but again why it makes better sense? 
* ##### how to show the reasoning formally ?
"""

# ╔═╡ 82b15590-8be9-4570-aad9-46f8133236ec
md"""

## Some motivating examples - (ML - classification)
#### Classification:  $Y$ is categorical

$\Large P(Y = \text{class 1} | X=x) = \sigma(x)$

* ##### the probability that $Y$ is class 1
"""

# ╔═╡ 382c174d-d6cd-43d1-8c1a-725687f971ee
TwoColumnWideRight(
md"""
\


!!! note "Question"
	##### How to classify or predict those _marked locations_?

""",

	Resource(bayes_figure_url * "logis_data.png", :width=>600)

	
)

# ╔═╡ a17437dc-b951-4337-b186-6d55849fe464
md"""

## Some motivating examples -  (ML - classification)

> ### _Which one do you prefer?_
"""

# ╔═╡ 58624ea1-383d-4a28-b9d2-7a84583ddaa1
TwoColumn(Resource(bayes_figure_url * "freq_logis.png"), Resource(bayes_figure_url * "bayes_logis.png"))

# ╔═╡ c8637533-4413-4361-8361-5242b33e4cde
md"""
#### The right one provides better predictions
* ##### but again why it makes better sense? 
* ##### how to show the reasoning formally ?
"""

# ╔═╡ 6294fe86-2097-403d-bc96-1a4655ef128d
# md"""
# ## More examples

# ### Some not-so-obvious uncertainty reasoning examples


# !!! example "Example (location vs food)"
# 	##### Why restaurants at *touristy locations* are usually *bad*?


# * ##### but why? how to justify it without only relying on intuition?
# \

# !!! example "Example (skills vs attractiveness)"
# 	##### Why *attractive* actors/actress are usually *not very skilled* ?


# * ##### again, how to formalise the reasonings?
# """

# ╔═╡ a3873cb6-67f1-41bf-b457-b90439868b16
md"""

## So far so good ...
\

#### Your intuition seems working well


\

#### Why we need _formal justification_ (or probability theory)?

* ##### your intuitive can fail miserably
  * turns out human brains are not wired to do uncertainty correctly all the time


* ##### if something cannot be justified with maths, then it's wobbly, or at least not very scientific
  * it applies to all science subjects


"""

# ╔═╡ d2b9f110-f475-42d4-8996-1db6dc57b91d
md"""

## Let's see some bad examples




!!! note "Positive or negative"
	#### You are tested positive for a rare cancer. 
	* ##### the test is highly accurate (99% accurate)

	#### Do you really have cancer?


"""

# ╔═╡ d34bb337-982b-45b5-ab61-512822a991de
Foldable("Answer", md"""

#### It depends. If the cancer is really rare, you are fine. Only around 10-ish% chance!

""")

# ╔═╡ e7cd47b9-8fbd-4291-a97e-b75b232dd91c
md"""
## Let's see more bad *examples*


"""

# ╔═╡ 53b602dc-979c-420e-bd42-9147d70730d8
md"""##### One of the great miscarriages of justice in modern British legal history"""

# ╔═╡ 681aa669-fb83-4f9d-a025-80b4d9cdc8f7
TwoColumn(md"""

### Sally Clark's case


- ##### a mother of two

- ##### both two children died: 
  - ##### sudden infant death syndrome (SIDS)
  - ##### a rare disease with a chance of $1/8543$
- ##### Sally Clark was convicted of murder 

""", html"""<br><br><center><img src="http://newsimg.bbc.co.uk/media/images/38744000/jpg/_38744329_clark300.jpg" width="300"/></center>""")

# ╔═╡ e439723e-5ae5-48dd-bbfe-78a0ff6b8413
md"""

##

#### *A key argument in  court*: a paediatrician professor testified 

> ##### *The chance of two children from an affluent family suffering cot death was 1 in 73 million*, *i.e.*
> ```math
> \Large \frac{1}{8543} \times \frac{1}{8543}
> ```
- ##### the jurors were swayed by the professor and gave a guilty verdict in 1999


#### *Aftermaths:* the uncertainty reasoning turned out to be flawed in all aspects
- ##### Sally Clark exonerated but died sadly in 2007 at the age of 42

"""

# ╔═╡ cc3d2d19-81f8-4f43-be30-81fc2680d1c7
md"""

## We are going to see 



#### How *Probability Theory* solves all above problems via one formula



$$\huge P(Query| Evidence)$$


* ##### *reasonable* inference results emerge automatically without ad hoc remedies




"""

# ╔═╡ 0d104033-945d-4c0d-ad1d-1a3750c9adea
# md"""

# !!! important ""
# 	##### ``\;\;\;\;\;\;`` _Probability theory_ is the tool to deal with **_Uncertainty_** correctly

# """

# ╔═╡ 7d8ed44a-06ba-4432-8345-55bb31eb8f1d
# md"""

# ## Prediction is uncertain -- regression


# """

# ╔═╡ 9e27864c-f5cb-4780-bcd3-e3d29a69742a
# Foldable("", md"""

# Prediction with a probabilistic distribution:

# ```math
# \large
# p(y_{test}|x_{test})
# ```
# """)

# ╔═╡ 5bd15469-e888-4505-a53d-49fef3329ea4
# md"Add linear regression: $(@bind add_lin CheckBox(default=false)),
# Add other fits: $(@bind add_gp CheckBox(default=false)),
# Add interval: $(@bind add_intv CheckBox(default=false))"

# ╔═╡ 3b4a2f77-587b-41fd-af92-17e9411929c8
# using GaussianProcesses

# ╔═╡ c9e0eaae-b340-434e-bdc9-dfdbc747221e
# let
# 	Random.seed!(123)
# 	# Generate random data for Gaussian process
# 	nobs = 4
# 	x = [0.5, 1.5, 4.5, 5.6]
# 	f(x) =  .75 * x + sin(x)
# 	y = f.(x) + 0.01 * rand(nobs)
	
# 	# Set-up mean and kernel
# 	se = SE(0.0, 0.0)
# 	m = MeanZero()
	
# 	# Construct and plot GP
# 	gp = GP(x, y, m, se, -1e5)

# 	plt = plot(x, y, st=:scatter, label="", markershape=:circle, markersize= 8,  xlabel=L"x", ylabel=L"y")
# 	xs = 0:0.05:2π
# 	plot!(xs, x -> f(x), color=:blue, lw=2, label="true function")
# 	# plot(gp;  xlabel=L"x", ylabel=L"y", title="Gaussian process", legend=false, xlim =[0, 2π])
	
# 	samples = rand(gp, xs, 10)
# 	w0, w1 = [ones(4) x] \ y

# 	if add_lin
# 		plot!(xs, (x) -> w0 + w1*x, lw=2, lc=:gray, label="")
# 	end
# 	if add_gp
# 		plot!(xs, samples, lw=2, label="", alpha=0.9)

# 		if add_intv
# 			plot!(gp; obsv=false, label="estimation mean")
# 		end
# 	end

# 	plt
# end

# ╔═╡ ff61cd9d-a193-44b3-a715-3c372ade7f79
md"# Probability theory"

# ╔═╡ 443fa256-ee34-43c0-8efd-c12560c00492
md"""

## What is probability? -- Frequentist 


#### There are two views: **Frequentist** and **Bayesian**

> #### **Frequentist's** 
> *  ##### *long-term stable frequency* of something (*i.e.* an **event**) happens
> ```math
> \large
> \mathbb{P}(E) = \lim_{n\rightarrow \infty} \frac{n(E)}{n}
> ```
> * ##### ``n(E)``: the number of times that event ``E`` happens
> * ##### ``n``: the total number of experiments

##



"""

# ╔═╡ 89df7ccb-53da-4b96-bbb4-fe39109467dd
md"Experiment times ``n``: $(@bind mc Slider(1:100000; show_value=true))"

# ╔═╡ bce5c041-be39-4ed1-8935-c389293400bc
penny_image = load(download("https://www.usacoinbook.com/us-coins/lincoln-memorial-cent.jpg"));

# ╔═╡ db6eb97c-558c-4206-a112-6ab3b0ad04c8
begin
	head = penny_image[:, 1:end÷2]
	tail = penny_image[:, end÷2:end]
end;

# ╔═╡ b742a37d-b49a-467b-bda7-6d39cce33125
TwoColumn(md"


#### Example: coin tossing

```math
\large
\begin{align}\mathbb{P}(\texttt{Head}) &=0.5 \\ 
&= \lim_{n\rightarrow \infty}\frac{n(\texttt{head})}{n}
\end{align}
```
* ##### `Head`: a head turns up
#### *Frequentist's* interpretation
* ##### toss the coin ``n \rightarrow \infty`` times, 
* ##### half of them will be head ", [head, tail])

# ╔═╡ ae5c476b-193e-44e3-b3a6-36c8944d4816
begin
	Random.seed!(3456)
	sampleCoins = rand(["head", "tail"], 100000);
	coin_exp = (sampleCoins .== "head")
	nhead = sum(coin_exp[1:mc])
end;

# ╔═╡ f44b5a95-95b5-4d88-927a-61d91ed51c53
sampleCoins[1:mc];

# ╔═╡ 221b8f09-6727-4613-8b96-02d70d337280
L"\large P(\texttt{head}) = \lim_{n\rightarrow \infty}\frac{n(\texttt{head})}{n} \approx \frac{%$nhead}{%$mc} = %$(round(nhead/mc; digits=3))"

# ╔═╡ 340c9b5b-5ca0-4313-870e-912d5c2dc451
let
	p_mc = nhead/mc
	plot(["tail", "head"], [1-p_mc, p_mc], st=:bar, label="", title=L"P(\texttt{Head})",ylabel="Probability", xtickfontsize=15,ylim = [0, 1.1])
end

# ╔═╡ deedb2db-8483-4026-975f-3d5af5a249b7
md"""
## What is probability ?  -- Bayesian

> #### **Bayesian's** interpretation
> * ##### *subjective* *belief* on something uncertain 
> * ##### as quantification of a personal belief
> * ##### no need to fixate to some repeated trials/experiments


#### For Sally's murder case,

```math
\Large	\mathbb{P}(\texttt{Guilty}) = 0.5
``` 

* ##### _subjectively_, *no preference* in terms of Sally being guilty or not


* ##### no need to assume there are millions of Sallys living in parallel universes or Sally lives her live million times

##


#### Both **interpretations** are valid and useful 

* ##### and it is more a philosophycal debate 


* ##### for many cases, the Bayesian interpretation is more natural
  * what is the chance of Brexit?
  * how can we travel back in time to 2018 again and again, infinite times?
"""

# ╔═╡ 128b6ad8-aa21-4d0a-8124-2cf433bc79c4
md"""

## Random variable 


### ``\cancel{\textbf{Random}}`` variable
\

> #### Variable ``\large X= 5``

* ##### ``X``: a _deterministic_ variable 
* ##### or *with ``100\%`` certainty*, ``X`` takes the value of 5


##


### Random variable: $X$
* ##### intuitively, the value is _*random*_ rather than _*certain*_


\

#### *For example*, $X$: the rolling realisation of a 6--faced die 🎲
* ##### the domain of possible values: ``X \in \{1,2,3,4,5,6 \}``


"""

# ╔═╡ 67f1a8a2-e78a-4220-b127-ae30385822d5
md"""

* #### and ``X`` has a *probability distribution* ``P(X)``
```math
\large
\begin{equation}  \begin{array}{c|cccccc} 
 & X = 1 & X = 2 & X = 3 & X = 4 & X = 5 & X = 6 \\
\hline
P(X) & 1/6 & 1/6 & 1/6 & 1/6 & 1/6 & 1/6

\end{array} \end{equation} 

```

"""

# ╔═╡ 403af436-d7f2-43c0-803a-8104ba69fcfd
md"""
## Probability distributions


#### Random variables' uncertainty is quantified by *probability distributions*

$$\Large P(X=x) \geq 0, \forall x\;\; \text{and}\;\; \sum_x{P(X=x)}=1$$
* ##### non-negative and sum to one


#### *Temperature* $T: P(T)$

```math
\begin{equation}  \begin{array}{c|c} 
T & P(T)\\
\hline
hot & 0.5 \\
cold & 0.5
\end{array} \end{equation} 
```


#### *Weather* $W: P(W)$

```math
\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```
"""

# ╔═╡ 3936846c-52e9-47ab-b557-38081dfdbe12
Foldable("Summation notation Σ", md"""

We use ``\sum`` to denote the sum of a series. For example, given vectors $\mathbf{x} = [x_1, x_2, \ldots, x_n]$,

The total sum is denoted as 

$$\sum_{i=1}^n x_i = x_1+x_2+\ldots+x_n$$


The total sum of the odd indexed entries can be loosely written as 

$$\sum_{i=1,3,\ldots} x_i = x_1+x_3+\ldots$$

Alternatively, to be more concrete, 

$$\sum_{i \in \{1,\ldots, n\}\; \&\; i \% 2=1}  x_i$$

Or one more way, note that $\mathbb{1}(\cdots)$ is an indicator function that returns either 1 or 0 based on the test condition of the input.

$$\sum_{i \in \{1,\ldots, n\}}  {\mathbb{1}(i\%2=1)} \cdot x_i$$

""")

# ╔═╡ 4d99c216-e32f-43a3-a122-ccbb697711fc
md"""
## Domain of a random variable


### *Discrete random variables* (main focus of CS3105)
* #### ``R:`` Is it raining?
  * ``\Omega =\{t, f\}``


* #### ``T:`` Is the temperature hot or cold?
  * ``\Omega =\{hot, cold\}``
\

### *Continuous random variables*

* #### ``T \in (-\infty, +\infty)`` the temperature in the room
  * ``\Omega = (-\infty, \infty)``



* #### ``D \in [0, +\infty)`` How long will it take to drive to St Andrews?
  * ``\Omega = [0, \infty)``



"""

# ╔═╡ 5b500acf-7029-43ff-9835-a26d8fe05194
md"""
## Notation



### We denote random variables with **Capital letters**

- #### ``X,Y, \texttt{Pass}, \texttt{Weather}`` are random variables


### Their realisations (values in ``\Omega``) in small cases
- #### ``x,y, +x, -y, \texttt{true, false, cloudy, sunny}`` are particular values r.v.s can take  


## Notation (cont.)

### Notation: $P(x)$  is a shorthand notation for $P(X=x)$

- #### so ``P(X)`` is assumed to be a distribution, but ``P(x)`` is a number

\


#### Therefore, $P(W)$ means the full distribution

```math
\large
\begin{equation}  \begin{array}{c|c} 
W & P(W)\\
\hline
sun & 0.6 \\
rain & 0.1 \\
fog & 0.2 \\
snow & 0.1
\end{array} \end{equation} 
```

* ##### while ``P(sun)`` is a number

```math
\large P(sun) = P(W=sun) = 0.6 
```


"""

# ╔═╡ 4d281b64-f9fb-43c3-81de-54b4e7761d6e
aside(tip(md"""
Short-hand notation

$$P(hot) = P(T=hot)$$

$$P(cold) = P(T=cold)$$


$$P(rain) = P(W=rain)$$

$$\vdots$$
"""))

# ╔═╡ 4bf768de-833f-45bf-9429-4820ff61553f
# md"""

# ## Examples of r.v.s

# | Variable  | Discrete or continous| Domain ``\, \Omega`` |
# | :---|:---:|:---:|
# | Toss of a coin | Discrete | ``\{0,1\}`` |
# | Roll of a die | Discrete |``\{1,2,\ldots, 6\}`` |
# | Outcome of a court case | Discrete |``\{0,1\}`` |
# | Number of heads of 100 coin tosses| Discrete|``\{0,1, \ldots, 100\}`` |
# | Number of covid cases | Discrete|``\{0,1,\ldots\}`` |
# | Height of a human | Continuous |``\mathbb{R}^+=(0, +\infty)`` |
# | The probability of coin's bias ``\theta``| Continuous|``[0,1]`` |
# | Measurement error of people's height| Continuous|``(-\infty, \infty)`` |
# """

# ╔═╡ 656da51f-fd35-4e89-9af5-b5f0fdf8618f
md"""
##  Binary random variable -- _a.k.a_ Bernoulli 




"""

# ╔═╡ 80038fee-b922-479d-9687-771e7e258fcf
md"Model parameter ``\theta``: $(@bind θ Slider(0:0.1:1, default=0.5; show_value=true))"

# ╔═╡ 7c03a15f-9ac1-465f-86a4-d2a6087e5970
TwoColumn(md"""

#### ``X`` taking binary values ``\{1, 0\}``: known as Bernoulli

```math
\large
\begin{equation}  \begin{array}{c|c} 
X & P(X)\\
\hline
1 & \theta \\
0 & 1-\theta
\end{array} \end{equation} 
```



* ##### e.g. coin tossing, rain or not, guilty or not
* ##### ``\theta`` (``0\leq \theta \leq 1``) is the parameter, called bias

##### Note that 
		  
$\large 0 \leq P(x) \leq 1; \text{and}\; \sum_{x=0,1}P(x) = \theta + 1-\theta = 1$


""", 

	begin
		bar(Bernoulli(θ), xticks=[0,1], xlabel=L"X", ylabel=L"P(X)", label="", ylim=[0,1.0], size=(250,300), title="Bernoulli dis.")
	end
)

# ╔═╡ e28e8089-f52b-440a-9861-895f9c378c84
# md"""
# ## Discrete r.v. -- Bernoulli 
# Probability distribution in one--line

# ```math
# \large 
# \begin{align}
# P(X=x) &=\begin{cases}\theta & x= 1 \\ 1-\theta & x=0 \end{cases} \\

# &=\boxed{ \theta^{x} (1-\theta)^{1-x}}
# \end{align}
# ```

# ``\text{for}\; x\in \{0, 1\}``
# * ``x=0``: ``P(X=0) = \theta^{0} (1-\theta)^{1-0} = \underbrace{\theta^0}_{1}\cdot (1-\theta)= 1-\theta``
# * ``x=1``: ``P(X=1) = \theta^{1} (1-\theta)^{1-1} = \theta\cdot (1-\theta)^0= \theta``

# """

# ╔═╡ 1e52d388-1e8d-4c20-b6e7-bcdd674ea406
md"""
## Categorical random variable




"""

# ╔═╡ b662605e-30ef-4e93-a71f-696e76e3ab45
TwoColumn(md"""


#### ``X`` takes categorical values

* ##### more than two choices

\

#### For example, alphabet usage in English 
- ##### ``\large X\in \{a,b,c\ldots, z, \_\}``
- #####  $\_$, $\texttt{e, i, n, o}$ are more likely to be used than e.g. letter $\texttt{z}$

""", html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure21.png" height="450"/></center>
""")

# ╔═╡ 59a38e32-c2f3-465f-928d-c05f8d69f496
md"""

## Categorical random variable 


#### For example, image classification


```math
\large
P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```

"""

# ╔═╡ a7a24713-a29a-4f0c-996b-f98305bac09c
md"""

## Categorical random variable 


#### For example, image classification


```math
\large
P(y_{test}|\mathbf{x}_{test}) = \begin{bmatrix}
\cdot \\
\cdot\\
\cdot\\
\end{bmatrix}
\begin{array}{l}
\texttt{cat}\\
\texttt{dog}\\
\texttt{others}\\
\end{array}
```

"""

# ╔═╡ 2ce6c56b-733c-42e8-a63b-d774cb6c199c
md"""

##
"""

# ╔═╡ 2e4df75b-0778-4ed4-840a-417da2d65204
md"""

##
"""

# ╔═╡ c5be7eb8-e0b6-48cc-8dbe-788fa6624999
Hs_catdogs = ["Cat", "Dog", "Others"];

# ╔═╡ 81ab9972-07bc-4ce9-9138-3359d4e34025
plt1, plt2, plt3=let
	ps = [.9, .05, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_cat = plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200), dpi=300)


	ps = [.05, .9, .05]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dog=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200),dpi=300)


	ps = [.25, .25, .5]
	texts = [Plots.text(L"%$(p)", 10) for (i, p) in enumerate(ps)]
	plt_dontknow=plot(ps, fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", color =:orange,  texts = texts,size=(200,200),dpi=300)

	plt_cat, plt_dog, plt_dontknow
end;

# ╔═╡ e8fd61f1-33a6-43d8-8056-fb7cf97291b5
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```
$(plt1)

""", md"""

""",
	md"""

""")

# ╔═╡ fc9e9bb6-2287-46c8-8518-c9d0804c094e
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""


""")

# ╔═╡ 8730b9a2-a1b4-456c-974c-ecd8880e6834
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```
$(plt3)

""")

# ╔═╡ dc8a3e36-2021-42dd-bc49-0eb6ab784fac
ThreeColumn(md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/cat1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt1)
""", md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/dog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{classify}
```

$(plt2)
""",
	md"""

$(Resource("https://leo.host.cs.st-andrews.ac.uk/figs/CS5914/dogcats/catdog1.png", :height=>200, :align=>""))

```math
\Big\Downarrow\;\; \texttt{???}
```

$(plot([.0, .0, .0], fill = true, st=:bar, xticks=(1:3, Hs_catdogs),  ylim =[0, 1.0], label="", title="", size=(200,200), dpi=300))
""")

# ╔═╡ 8a5f2129-4cf5-45d3-8e05-3416b6d8e6fd
md"""

## Specify probability distributions


#### _So far_, distributions are represented as (look-up) tables

* ##### for example, 

```math
\Large
\begin{equation}  \begin{array}{c|c} 
X & P(X)\\
\hline
1 & 0.8 \\
0 & 0.2
\end{array} \end{equation} 
```


* ##### or equivalently

$$\Large P(X) = \begin{cases} 0.8 & x=1 \\
0.2 & x=0\end{cases}$$ 


### _But they do not have to be look-up tables_ 
* ##### distributions are just functions
"""

# ╔═╡ 556617f4-4e88-45f4-9d91-066c24473c44
md"""

## Example -- Binomial random variable


#### Toss a coin with bias (``0\leq \theta \leq 1``) independently ``n`` times: 

```math
\Large  \{Y_1, Y_2, \ldots, Y_n\} \in \{0,1\}^n
```

```math
\Large
P(Y_i) =\begin{cases}\theta &  Y_i=1 \\ 1-\theta & Y_i=0 \end{cases}
```


#### The *number of heads* (or ``1``s) (or the sum) is 

$$\Large X= \sum_{i=1}^n Y_i$$

## Example -- Binomial random variable


#### Toss a coin with bias (``0\leq \theta \leq 1``) independently ``n`` times: 

```math
\Large  \{Y_1, Y_2, \ldots, Y_n\} \in \{0,1\}^n
```

```math
\Large
P(Y_i) =\begin{cases}\theta &  Y_i=1 \\ 1-\theta & Y_i=0 \end{cases}
```


#### The *number of heads* (or ``1``s) (or the sum) is 

$$\Large X= \sum_{i=1}^n Y_i$$

####  And X is a Binomial random variable 

$$\Large P(X=x) = \text{Binom}(X; n,\theta)= \binom{n}{x} (1-\theta)^{n-x} \theta^{x},$$


* #### domain: ``x\in \{0,1,\ldots, n\}``

 
* #### ``\Large\binom{n}{x}`` is the *binomial coefficient*


* #### the distribution is specified as a function 
  * ##### still satisfies: ``P(x) \geq 0`` and ``\sum_x P(x)=1``

"""

# ╔═╡ 4d6badcc-c061-4e63-a156-167376f131eb
md"Bias of each trial ``\theta`` $(@bind θ_bi Slider(0:0.05:1, default=0.7, show_value=true)) , Total trials: ``n`` $(@bind nb Slider(2:1:20, default=10, show_value=true)),
"

# ╔═╡ c3910dd8-4919-463f-9af0-bc554566c681
let
	binom = Binomial(nb, θ_bi)
	bar(binom, label="", xlabel=L"X", xticks = 0:nb, ylabel=L"P(X)", title="Binomial with: "*L"n=%$(nb),\;\;\theta = %$(θ_bi)",legend=:topleft, framestyle=:semi)

end

# ╔═╡ f664e72d-e762-4dea-ba11-bc8c6b2863f9
# md"""

# ## Discrete r.v. -- Binomial (conti.)


# !!! question "Question"
# 	Toss a coin (with bias ``0\leq \theta \leq 1``)  ``n`` times
# 	* what is the probability the number of heads is more than the half?

# We can use the Binomial distribution:

# ```math
# \large
# P(X \geq \lfloor n/2\rfloor + 1) = \sum_{x \geq \lfloor n/2\rfloor + 1}P(x)
# ```

# * recall ``X`` is the total number of heads
# """

# ╔═╡ c134a0b6-8754-48bd-a2ca-932542744407
# let
# 	binom = Binomial(nb, θ_bi)
# 	bar(binom, label="", xlabel=L"X", xticks = 0:nb, ylabel=L"P(X)", title=L"n=%$(nb),\;\;\theta = %$(θ_bi)", legend=:topleft)

# 	p_more_heads = exp(logsumexp(logpdf.(binom, floor(nb/2)+1:1:nb)))
# 	vspan!([floor(nb/2)+0.5, nb+0.5], alpha=0.5, label=L"P(X \geq \lfloor n/2 \rfloor +1)\approx%$(round(p_more_heads; digits=2))", framestyle=:semi)
# end

# ╔═╡ 0cffc249-5269-4322-aa4a-ccf7075ec8e1
# aside(tip(md"""

# We can add probabilities together if the events are **disjoint**: the sum is the union of all event.

# """))

# ╔═╡ e557ad8b-9e4f-4209-908f-2251e2e2cde9
# md"""
# ## Joint distribution


# A **joint distribution** over a set of random variables: ``X_1, X_2, \ldots, X_n`` 


# ```math
# \large
# \begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
# ```

# * the joint event ``\{X_1=x_1, X_2=x_2, \ldots, X_n=x_n\}``'s distribution

# * must still statisfy

# $P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 

# ## Joint distribution -- examples


# For example, joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
# ```math
# \begin{equation}
# \begin{array}{c c |c} 
# T & W & P\\
# \hline
# hot & sun & 0.4 \\
# hot & rain & 0.1 \\
# cold & sun  & 0.2\\
# cold & rain & 0.3\end{array} \end{equation} 
# ```



# """

# ╔═╡ 42d137f0-7f7e-4081-ba91-d16c8c8d8d62
md"""
## Joint distributions


### A *joint distribution* over a set of random variables 

$\Large X_1, X_2, \ldots, X_n$ 

```math
\Large \begin{equation} P(X_1= x_1, X_2=x_2,\ldots, X_n= x_n) = P(x_1, x_2, \ldots, x_n) \end{equation} 
```

* ##### the probablity of the set being true at the same time
* ##### still must statisfy

$\large P(x_1, x_2, \ldots, x_n) \geq 0\;\; \text{and}\;\;  \sum_{x_1, x_2, \ldots, x_n} P(x_1, x_2, \ldots, x_n) =1$ 


##### Example: joint distribution of temperature (``T``) and weather (``W``): ``P(T,W)``
```math
\large \begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```


##### Sometimes, the same table can be formed as a flat one

```math
\large \begin{equation}
\begin{array}{c| c c} 
& sun & rain \\
\hline
hot & 0.4 & 0.1 \\
cold & 0.2 & 0.3

\end{array} \end{equation} 
```


"""

# ╔═╡ 3bc15a7d-affa-428e-b563-88abb05fe99c
# md"""
# ## Size of a joint distribution

# ```math
# \Large
# \begin{equation}
# \begin{array}{c  c |c} 
# T & W & P \\
# \hline
# hot & sun & 0.4 \\
# hot & rain & 0.1 \\
# cold & sun  & 0.2\\
# cold & rain & 0.3\end{array} \end{equation} 
# ```


# !!! question "Question" 
# 	#### How many *rows* are there in a joint distribution?
# """

# ╔═╡ 2c27858c-79b7-4b58-8cb9-22860c84174b
# Foldable("Joint distribution explodes as n", md"""

# * #### ``n`` variables with domain size ``d``: ``d^n`` (``d=2`` for this case)
# * #### the table size explodes when ``n`` is large!
# """)

# ╔═╡ 25c1c999-c9b0-437a-99f3-6bb59482ca7d
# md"""

# ## Joint distribution --  examples


# """

# ╔═╡ 1877ebc8-790f-483a-acf3-9288df9ee7cc
# TwoColumn(md"""
# \
# Bi-letter example: $X,Y$ represents the _first_ and _second_ letter
#   * *e.g.* $X = \texttt{s}, Y = \texttt{t}$, means a bi--letter "$\texttt{st}$"

# * there are $27 \times 27$ entries
# * very common bigrams are $\texttt{in, re, he, th}$, \_$\texttt{a}$ (starting with ``\texttt{a}``)
# * uncommon bigrams are $\texttt{aa, az, tb, j}$\_ (ending with ``\texttt{j}``)
# * sum all $27\times 27$ entries will be 1

# """, html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure19.png" width = "300" height="300"/></center>
# """)

# ╔═╡ 0ff13367-204c-43a7-a60b-1daa0df540ac
md"""
## Two probability rules
\


### Luckily, there are only two rules 

* #### *sum rule*
  * ##### from joint distribution to marginal distribution



* #### *product rule*
  * ##### from marginal and conditional to the joint



"""

# ╔═╡ ee656c24-6076-462a-9acb-da66628d61fb
md"""
## Probability rule 1: sum rule
\



$$\Large P(X_1) = \sum_{x_2}P(X_1, X_2=x_2);\;\; P(X_2) = \sum_{x_1}P(X_1=x_1, X_2),$$

* ##### ``P(X_1), P(X_2)`` are called *marginal probability distribution*

"""

# ╔═╡ 39e45305-dd18-477f-a972-b60e8fc9d4ae
md"""

## Sum rule: example


#### Sum rule allows us to find the marginal distribution

* ##### marginalisation (sum out): combine collapsed rows by adding
"""

# ╔═╡ 5df5d3ba-d6e8-4b8d-b7a9-008822a061ba
Resource(figure_url * "sumrule_.png", :width=>800, :align=>"left")

# ╔═╡ cb4dfdd2-a685-46ea-932f-7df1b1f0bc97
Foldable("Correction", md"""

The sum rule should have been
$$P(t) = \sum_w P(t, w)$$
and
$$P(w) = \sum_t P(t, w)$$		 
		 """)

# ╔═╡ 64360513-f68f-4ec0-b393-de147a4fa891
# md"Note it implies: $P(X_1= x_1) = \sum_{x_2} P(X_1= x_1, X_2=x_2)$
# * also known as total probability rule"

# ╔═╡ a198271a-9344-4751-a02f-cf64a80d7147
# md"""

# ## Sum rule: example 


# """

# ╔═╡ c26ee879-92b6-4d2b-819c-308db5e32ff4
# TwoColumn(md"""
# \
# Given bi-letter joint 

# $P(X, Y)$

# how to compute the marginals?

# * ``P(X)``


# * ``P(Y)``

# """, html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure19.png" width = "300" height="300"/></center>
# """)

# ╔═╡ 271009dc-c42b-40bc-9896-e6fc16238a73
md"""

## Conditional probability

### Conditional probability is defined as

$$\Large P(A=a|B=b) = \frac{P(A=a, B=b)}{P(B=b)}$$

* #### read: *probability of ``A`` given ``B``*
* #### the probability of $A=a$ given $B=b$ is true



"""

# ╔═╡ a01ebe11-dba2-45df-9fe3-1343576c2071
Resource(figure_url * "condiprob.png", :width=>800, :align=>"left")

# ╔═╡ f1274cf1-68b2-4a29-8cb3-0c73f5c1a4fc
md"""
##

#### *Example*
```math
\large\begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
 cold & sun  & 0.2\\
 cold & rain & 0.3\end{array} \end{equation} 
```

##### The conditional probability 

$$\large P(W=s|T=c) = \frac{P(W=s, T=c)}{\underbrace{P(T=c)}_{P(s, c) + P(r,c ) = 0.5}} = \frac{0.2}{0.2+0.3} = 0.4$$

"""

# ╔═╡ 73b617f0-3339-4d99-80b1-ec01106e3cba
md"""


#### Exercise: compute ``P(W=r|T=c)``
"""

# ╔═╡ 75e14d8e-18bc-4fa8-8dcd-b6f5d1c51424
Foldable("Solution", md"


$$P(W=r|T=c) = \frac{P(W=r, T=c)}{{P(T=c)}} = \frac{0.3}{0.2+0.3} = 0.6$$


Note that $P(W=r|T=c) + P(W=s|T=c) =1$
")

# ╔═╡ 67dd55c1-9a5b-4b33-877b-e12192f9c645
md"""

## Aside: normalisation and ``\propto``

### *"Normalisation"*: divide each term by the total sum
* #### examples

$\langle 0.2, 0.3 \rangle \stackrel{\text{normalise}}{\Longrightarrow} \left \langle \frac{0.2}{0.2+0.3}, \frac{0.3}{0.2+0.3} \right \rangle = \left \langle \frac{2}{5}, \frac{3}{5} \right  \rangle$

$\langle 2, 3 \rangle \stackrel{\text{normalise}}{\Longrightarrow} \left\langle \frac{2}{2+3}, \frac{3}{2+3} \right\rangle  = \left\langle \frac{2}{5}, \frac{3}{5} \right\rangle$



### ``\propto``: *proportional*

* #### ``\mathbf{x} \propto \mathbf{y}`` means, there is some constant ``\alpha >0`` such that

$$\langle x_1, x_2, \ldots x_n \rangle = \alpha \langle y_1, y_2, \ldots y_n \rangle$$

* #### *e.g.* $\langle 0.2, 0.3 \rangle  \propto \langle 2, 3 \rangle$, and ``\alpha = ?``

* #### in other words, they are equal after normalisation


"""

# ╔═╡ 74c1395f-a049-47ae-b9fe-4ed64e1debd1
md"""

## Conditional probabilities with ``\propto``


#### *Conditional probabilities* can be specified as

```math
\Large
P(A|B=b) \propto P(A, B=b)
```
\


#### *Therefore*, ``P(A|B=b)`` can be computed by two steps

* ##### *select* the rows based on the condition
* ##### *normalise* the filtered probabilities 




"""

# ╔═╡ c2bf647d-542c-47d8-87de-65a644eac516
md"""


#### _For example_, to compute $P(W|T=c)$
"""

# ╔═╡ 1607c49f-2ece-402a-9abb-6304f96c1dca
Resource(figure_url * "selectnorm.png", :width=>850, :align=>"left")

# ╔═╡ 3b857552-c6aa-44b2-8a56-6c2315bd9a51
md"""

#### Note the proportional ``\propto`` relationship 

$\Large P(W|T=c) \propto P(W, T=c)$

* #### *i.e.* ``\langle 0.4,0.6 \rangle \propto \langle 0.2, 0.3\rangle``

#### Therefore, after normalisation, the sum is one!

$\Large P(W=r|T=c) + P(W=s|T=c) =1$

* ##### it is a valid distribution of $W$
"""

# ╔═╡ 6004c880-974c-4fb2-897e-b3a0add22508
md"""

## Quiz


#### Caclulate $P(W|T=hot)$

```math
\Large \begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
hot & sun & 0.4 \\
hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```



"""

# ╔═╡ b2490239-a2ba-453e-9a6e-ffb42a1b32b8
Foldable("Solution", md"

Select + Normalisation

```math
\begin{equation}
\begin{array}{c c |c} 
T & W & P\\
\hline
\rowcolor{salmon} hot & sun & 0.4 \\
\rowcolor{salmon} hot & rain & 0.1 \\
cold & sun  & 0.2\\
cold & rain & 0.3\end{array} \end{equation} 
```


$$P(W=r|T=h) = \frac{P(W=r, T=h)}{{P(T=h)}} = \frac{0.1}{0.1+0.4} = 0.2$$


$$P(W=s|T=h) = \frac{P(W=s, T=h)}{{P(T=h)}} = \frac{0.4}{0.1+0.4} = 0.8$$


")

# ╔═╡ ddbdd564-750d-4728-b8e2-5f548cddbd05
md"""

## Conditional distribution

#### ``\large P(W|T=c)`` forms a *distribution* of ``W``

* #### the conditional probability distribution of $W$ given ``T=c``


```math
\Large \begin{equation}  \begin{array}{c|c} 
W & P(W|T=c)\\
\hline
sun & 0.4 \\
rain & 0.6
\end{array} \end{equation} 
```

* #### indeed, just check ``\sum_w P(W=w|T=c) = 1``


#### Similarly, the conditional distribution of W given ``T=hot``

$\Large P(W|T=hot)$

* ##### also a distribution of ``W`` (under a different condition)


"""

# ╔═╡ 0b1703b4-8a0e-4f3a-9a94-4a778ee451df
Resource(figure_url * "condiandjoint.png", :width=>800, :align=>"left")

# ╔═╡ 70b362fd-b3f2-4013-8582-93eddc844029
md"""

## Independence

### Random variable $X,Y$ are independent, if 

$$\Large \forall x,y : P(x,y) = P(x)P(y)$$

* ##### the joint is formed by the product of the marginals


"""

# ╔═╡ ffd5f874-58ff-4c7c-afa0-6cd4b2dce9af
md"""

## Independence

### Random variable $X,Y$ are independent, if 

$$\Large \forall x,y : P(x,y) = P(x)P(y)$$

* ##### the joint is formed by the product of the marginals



### Two equivalent definitions

$$\Large \forall x,y: P(x|y) = P(x)$$


$$\Large \forall x,y: P(y|x) = P(y)$$

* ##### intuition: knowing $Y$ (conditional on $Y=y$) has no effect on predicting ``X`` (or knowing $X$ has no effect on predicting $Y$)
  

#### $X,Y$ independent, denoted as ``X \perp Y``


## Independence

#### For *multiple* independent random variables,

```math
\Large P(X_1, X_2, \ldots, X_n) = \prod_i P(X_i)
```
"""

# ╔═╡ e83c3a60-1e85-4e07-895d-7a3a42cf1f7f
md"""

## Example


#### N independent coin flips

"""

# ╔═╡ 361ff521-a117-405f-ba57-91fc2ee00ce2
Resource(figure_url * "ncoinflips.png", :width=>800, :align=>"left")

# ╔═╡ 09967f8c-fbc0-4920-9ed6-48e007134026
md"""

#### The joint is ``P(X_1, X_2, \ldots, X_n)= \prod_i P(X_i)``
* #### due to independence assumption



```math
\large\begin{equation}
\begin{array}{c c c c |c} 
X_1 & X_2 &   \ldots & X_n & P\\
\hline
h & h & \ldots & h & 0.5^n\\
h & h & \ldots & t & 0.5^n \\
\vdots & \vdots & \ldots &  \vdots\\
t & t & \ldots & t & 0.5^n\end{array} \end{equation} 
```
"""

# ╔═╡ 62627b47-5ec9-4d7d-9e94-2148ff198f66
md"""
## Independence: genuine coin toss

> #### Two sequences of 300 “coin flips” (H for heads, T for tails). 
> 
> * #### which one is the genuine **independent** coin tosses?

#### **Sequence 1**

> 	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

#### **Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

* ##### both of them have ``N_h = 148`` and ``N_t= 152``
"""

# ╔═╡ fc09b97a-13c9-4721-83ca-f7caa5f55079
begin
	seq1="TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT"
	seq2 = "HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT"
	sequence1=map((x) -> x=='H' ? 1 : 2,  [c for c in seq1])
	sequence2=map((x) -> x=='H' ? 1 : 2,  [c for c in seq2])
end;

# ╔═╡ 95779ca4-b743-43f1-af12-6b14c0e28f0b
md"""

## Independence: genuine coin toss (cont.)


#### Recall **independence**'s definition

```math
\Large
P(X_{t+1}|X_{t}) = P(X_{t+1})
```

* ##### ``X_{t}``: the tossing result at ``t``
* ##### ``X_{t+1}``: the next tossing result at ``t+1``


#### And the conditional distribution should be (due to independence)

```math
\large
P(X_{t+1}=\texttt{h}|X_{t}=\texttt{h}) = P(X_{t+1}=\texttt{h}|X_{t}=\texttt{t}) =P(X_{t+1}=\texttt{h}) = 0.5
```

"""

# ╔═╡ 0f280847-2404-4211-8221-e30418cf4d42
md"""

##


#### **Sequence 1**

>	TTHHTHTTHTTTHTTTHTTTHTTHTHHTHHTHTHHTTTHHTHTHTTHTHHTTHTHHTHTTTHHTTHHTTHHHTHHTHTTHTHTTHHTHHHTTHTHTTTHHTTHTHTHTHTHTTHTHTHHHTTHTHTHHTHHHTHTHTTHTTHHTHTHTHTTHHTTHTHTTHHHTHTHTHTTHTTHHTTHTHHTHHHTTHHTHTTHTHTHTHTHTHTHHHTHTHTHTHHTHHTHTHTTHTTTHHTHTTTHTHHTHHHHTTTHHTHTHTHTHHHTTHHTHTTTHTHHTHTHTHHTHTTHTTHTHHTHTHTTT

#### The joint frequency table is

```math
\large
\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 46 & 102 \\ 

X_t= \texttt t & 102 & 49 \\ 

\end{array} \end{equation} 

```

* ``\large P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{46}{46+102} \approx 0.311 \ll 0.5``
* ``\large P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{102}{102+49} \approx 0.675 \gg 0.5``

"""

# ╔═╡ b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
md"""

##

#### **Sequence 2**

>	HTHHHTHTTHHTTTTTTTTHHHTTTHHTTTTHHTTHHHTTHTHTTTTTTHTHTTTTHHHHTHTHTTHTTTHTTHTTTTHTHHTHHHHTTTTTHHHHTHHHTTTTHTHTTHHHHTHHHHHHHHTTHHTHHTHHHHHHHTTHTHTTTHHTTTTHTHHTTHTTHTHTHTTHHHHHTTHTTTHTHTHHTTTTHTTTTTHHTHTHHHHTTTTHTHHHTHHTHTHTHTHHHTHTTHHHTHHHHHHTHHHTHTTTHHHTTTHHTHTTHHTHHHTHTTHTTHTTTHHTHTHTTTTHTHTHTTHTHTHT

```math
\Large
\begin{equation}  \begin{array}{c|cc} 
n(X_{t}, X_{t+1}) & X_{t+1} = \texttt h & X_{t+1} =\texttt t \\
\hline
X_t =\texttt h & 71 & 77 \\ 

X_t= \texttt t & 76 & 75 \\ 

\end{array} \end{equation} 

```

* ``\large P(X_{t+1}=\texttt h|X_t=\texttt h) =\frac{71}{71+77} \approx 0.48 \approx 0.5``
* ``\large P(X_{t+1}=\texttt h|X_t=\texttt t) =\frac{76}{76+75} \approx 0.503 \approx 0.5``
"""

# ╔═╡ 473ec1bb-4c5e-468b-b19d-335d77868d36
# md"""

# ## Independent or not ?
# """

# ╔═╡ 6f368af8-84fc-4075-9ee2-ca5740a81c4f
# TwoColumn(md"""
# \
# \
# \

# !!! question ""
# 	> Are two letters: ``X, Y`` **independent**?

# """, html"""<center><img src="https://leo.host.cs.st-andrews.ac.uk/figs/figure19.png" width = "300" height="300"/></center>
# """)

# ╔═╡ 4ba80e92-fa56-44f7-b86d-7eafdb70827f
# Foldable("Answer", md"""

# If independent, then for all ``x, y``

# $P(X=x|Y=y) = P(X=x)$

# or equivalently


# $P(Y=y|X=x) = P(Y=y)$ 



# * change the first letter ``x``, ``P(Y|X=x)`` do not change


# * it means 

# ```math
# P(Y|x = \texttt{a}) = P(Y|x = \texttt{b})
# ```

# * in other words, all the rows of ``P(Y|x)`` should be the same!

# * or all the columns of ``P(X|y)`` should be the same!



# ![https://leo.host.cs.st-andrews.ac.uk/figs/figure20.png](https://leo.host.cs.st-andrews.ac.uk/figs/figure20.png)
# """)

# ╔═╡ 0d8509fb-002c-4c4b-b228-f62bebd58c71
# md"""

# ## So if independent

# """

# ╔═╡ efc1bfe2-0981-47d9-8cef-7dc8168678ad
md"""


## Probability rule 2: product rule



### If ``X \perp Y`` (independent), then life is simple:


```math
\Large
P(X, Y) = P(X)P(Y)
```
\


#### When ``X, Y`` are *not* independent -- we use product rule (chain rule)


$$\Large P(X, Y) = P(X)P(Y|X);\;\; P(X, Y) = P(Y)P(X|Y)$$

  * ##### the chain order doesn't matter


  * ##### chain rule states how joint distribution factorised as a product
"""

# ╔═╡ 4b47f07a-7e3e-4769-8398-02776872784c
md"""

##

#### Chain rule is just *rearranged version* of conditional probability 

$$\Large P(X|Y) = \frac{P(X, Y)}{P(Y)} \Rightarrow P(X|Y)P(Y) = P(X, Y)$$


#### Similarly, 

$$\Large P(Y|X) = \frac{P(X, Y)}{P(X)} \Rightarrow P(Y|X)P(X) = P(X, Y)$$

"""

# ╔═╡ 2dd51262-6859-41c3-b0c4-8a17bc19c1de
md"""
## Example
"""

# ╔═╡ fa805bf5-c3b9-4e5e-b032-9b6961ba12c6
Resource(figure_url * "prodrule.png", :width=>800, :align=>"left")

# ╔═╡ acf37074-d5dc-4d7d-96ae-86690c497435
# md"""

# | ``D`` | ``W`` | ``P(D, W)``|
# | --- | --- | :---: |
# | wet| sun| ``0.8 \times 0.1 = 0.08`` |
# | dry| sun| |
# | wet| rain| |
# | dry| rain| |

# """

# ╔═╡ d6d93998-1e19-46b6-8a73-8403c1cb5557
aside(tip(md"""


```math
P(D, W) = P(W) P(D|W)
```
"""))

# ╔═╡ 098a2f02-b0ee-4c05-b023-f5a6303012dd
md"""
## Example -- product rule

"""

# ╔═╡ 36f9056d-3c77-4f2e-9201-7cd4c2c67ccf
Resource(figure_url * "/prodrule1.png", :width=>800, :align=>"left")

# ╔═╡ 1e6a71cf-dc14-45d4-a9c8-7fec8486ffe2
md"""
## Example -- product rule

"""

# ╔═╡ 7ef9a7d7-3d99-435c-86b5-8de0e0f1dbc1
Resource(figure_url * "/prodrule2.png", :width=>800, :align=>"left")

# ╔═╡ 48738f54-f621-4f73-809a-4ae3e4bcedbe
md"""
## Example -- product rule

"""

# ╔═╡ 3728a7d2-8c06-4a14-b2be-7af880be3040
Resource(figure_url * "/prodrule3.png", :width=>800, :align=>"left")

# ╔═╡ 6b9cd068-3faa-4c24-ac14-e30718cee656
md"""

## Product rule: generalisation


#### More generally, the product rule is valid for $n$ variables
\

#### for three
$\Large P(x_1, x_2, x_3) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)$

#### for any ``n``:

$\large P(x_1, x_2 \ldots x_n) = P(x_1)P(x_2|x_1)P(x_3|x_1,x_2) \ldots=\prod_{i=1}^n P(x_i|x_1, \ldots, x_{i-1})$

  * ##### it chains variables together: **chain rule**
"""

# ╔═╡ 8081bdc2-2d1f-445b-ace9-b8f0c378f279
Foldable("Proof", md"""
		 It is easy to prove the generalised product rule. For example, to prove

		 $\large P(x_1, x_2, x_3) = P(x_1)P(x_2|x_1)P(x_3|x_1, x_2)$

		 $$\begin{align}\text{R.H.S} &= P(x_1)P(x_2|x_1)P(x_3|x_1, x_2) \\ 
		 
		 &= P(x_1)\frac{P(x_2, x_1)}{P(x_1)}P(x_3|x_1, x_2)\; \;\;\;\text{definition of conditional prob.}\\ 

		 &= \cancel{P(x_1)}\frac{\cancel{P(x_2, x_1)}}{\cancel{P(x_1)}} \frac{P(x_3, x_1, x_2)}{\cancel{P(x_1, x_2)}} \\ 
		 &= P(x_3, x_1, x_2) \\

		 &= P(x_1, x_2, x_3) = \text{L.H.S.}
		 \end{align}$$

		 Remember that for joint distribution, the event $(x_3, x_1, x_2)=(X_1 =x_1 \land X_2=x_2 \land X_3=x_3)$, therefore the last equality holds. 
		 """)

# ╔═╡ 2302dedd-bafc-45a9-b39f-48c74ed2339d
md"""
## Product rule: generalisation


### You can chain ``\{X_1, \ldots, X_n\}`` in _any_ order


* #### *e.g.* reverse order 

$\large P(x_1, x_2, x_3) = P(x_3)P(x_2|x_3)P(x_{1}|x_2,x_{3})$

* #### or an arbitrary order

$\large P(x_1, x_2, x_3) = P(x_2)P(x_3|x_2)P(x_{1}|x_2,x_{3})$


"""

# ╔═╡ 5ef47e4c-3658-4a28-90e6-a30f7ed95ea6
md"""

## Relevant chapters

\

* #### Chapter 13. Quantifying Uncertainty


* #### Chapter 14. Probabilistic Reasoning
"""

# ╔═╡ 0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
md"""

## Appendix
"""

# ╔═╡ 8687dbd1-4857-40e4-b9cb-af469b8563e2
function perp_square(origin, vx, vy; δ=0.1) 
	x = δ * vx/sqrt(norm(vx))
	y = δ * vy/sqrt(norm(vy))
	xyunit = origin+ x + y
	xunit = origin + x
	yunit = origin +y
	Shape([origin[1], xunit[1], xyunit[1], yunit[1]], [origin[2], xunit[2], xyunit[2], yunit[2]])
end

# ╔═╡ fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
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

# ╔═╡ 4d1ca0d2-9eb0-4166-834c-1641e8bd5a9e
begin
	function true_f(x)
		-5*tanh(0.5*x) * (1- tanh(0.5*x)^2)
	end
	Random.seed!(100)
	x_input = [range(-6, -1.5, 10); range(1.5, 6, 8)]
	y_output = true_f.(x_input) .+ sqrt(0.05) * randn(length(x_input))
end;

# ╔═╡ 97c42357-55bd-4805-8df9-cf42c6bf5243
begin
	scatter(x_input, y_output, label="Observations", xlim=[-8, 8], ylim=[-3, 3], xlabel=L"X", ylabel=L"Y", ratio=0.8, size=(700,300))	
	# plot!(true_f, lw=2, xlim=[-10, 10], framestyle=:default,  lc=:gray, ls=:dash,label="true function", xlabel=L"x", ylabel=L"y")
end

# ╔═╡ 9ac8dd17-0319-46f0-aba4-89b5ce082859
plt1_ = let
	scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")	
	plot!(true_f, lw=2, framestyle=:default,  lc=:red, ls=:solid,label=L"f(x)", xlabel=L"x", ylabel=L"y")
end;

# ╔═╡ 862b0317-f766-47b9-9e57-167f14d534f8
begin
	function poly_expand(x; order = 2) # expand the design matrix to the pth order
		n = length(x)
		return hcat([x.^p for p in 0:order]...)
	end
end

# ╔═╡ 4bfb863d-c5f9-4ecb-a946-fde7a2148446
begin
	Random.seed!(123)
	w = randn(length(x_input)) ./ 2
	b = range(-18, 22, length(x_input)) |> collect
	Φ = tanh.(x_input .- w' .* b')
end;

# ╔═╡ 279aee95-386a-4c77-81c5-8586400e93b2
begin
	poly_order = 6
	# Φ = poly_expand(x_input; order = poly_order)
	freq_ols_model = lm(Φ, y_output);
	# apply the same expansion on the testing dataset
	x_test = -10:0.1:10
	Φₜₑₛₜ = tanh.(x_test .- w' .* b')
	tₜₑₛₜ = true_f.(x_test)
	# predict on the test dataset
	βₘₗ = coef(freq_ols_model)
	pred_y_ols = Φₜₑₛₜ * βₘₗ 
end;

# ╔═╡ 63898b4b-b5a5-42bd-aa29-08088208a51b
plt2_ = let
	# plot(x_test, tₜₑₛₜ, linecolor=:black, ylim= [-3, 3], lw=2, linestyle=:dash, lc=:gray,framestyle=:default, label="true signal")
	scatter(x_input, y_output, label="", xlim=[-13, 13], ylim=[-5, 5], xlabel=L"x", ylabel=L"y")	
	plot!(x_test, pred_y_ols, linestyle=:solid, lc =:blue, lw=2, xlabel=L"x", ylabel=L"y", legend=:topright, label=L"f(x)")
end;

# ╔═╡ 61259c61-2c9b-4ead-aa72-b3ecdc21d90b
plt = plot(plt1_, plt2_, layout=(1,2), size=(600,300));

# ╔═╡ 0774a6b6-d620-4e02-959f-817a5c13523f
md"""
##


!!! note "Question"
	#### Which function below is more reasonable?
	
	* ##### the red one: $\textcolor{red}{f(x)}$
	* ##### the blue one: $\textcolor{blue}{f(x)}$

$(begin 
	plt
end)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
Images = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
Distributions = "~0.25.122"
GLM = "~1.9.0"
HypertextLiteral = "~0.9.5"
Images = "~0.26.2"
LaTeXStrings = "~1.4.0"
Latexify = "~0.16.10"
LogExpFunctions = "~0.3.29"
Plots = "~1.41.1"
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.73"
StatsBase = "~0.34.7"
StatsPlots = "~0.15.8"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "837555e51b05fc5d86eba3954d6db2b509a199e9"

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

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "d81ae5489e13bc03567d4fbbb06c546a5e53c857"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.22.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = ["CUDSS", "CUDA"]
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceMetalExt = "Metal"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "4126b08903b777c88edf1754288144a0492c05ad"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.8"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.BitTwiddlingConvenienceFunctions]]
deps = ["Static"]
git-tree-sha1 = "f21cfd4950cb9f0587d5067e69405ad2acd27b87"
uuid = "62783981-4cbd-42fc-bca8-16325de8dc4b"
version = "0.1.6"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1b96ea4a01afe0ea4090c5c8039690672dd13f2e"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.9+0"

[[deps.CEnum]]
git-tree-sha1 = "389ad5c84de1ae7cf0e28e381131c98ea87d54fc"
uuid = "fa961155-64e5-5f13-b03f-caf6b980ea82"
version = "0.5.0"

[[deps.CPUSummary]]
deps = ["CpuId", "IfElse", "PrecompileTools", "Preferences", "Static"]
git-tree-sha1 = "f3a21d7fc84ba618a779d1ed2fcca2e682865bab"
uuid = "2a0fbf3d-bb9c-48f3-b0a9-814d99fd7ab9"
version = "0.2.7"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "fde3bf89aead2e723284a8ff9cdf5b551ed700e8"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.5+0"

[[deps.CatIndices]]
deps = ["CustomUnitRanges", "OffsetArrays"]
git-tree-sha1 = "a0f80a09780eed9b1d106a1bf62041c2efc995bc"
uuid = "aafaddc9-749c-510e-ac4f-586e18779b91"
version = "0.2.2"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "e4c6a16e77171a5f5e25e9646617ab1c276c5607"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.26.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.ChunkCodecCore]]
git-tree-sha1 = "51f4c10ee01bda57371e977931de39ee0f0cdb3e"
uuid = "0b6fb165-00bc-4d37-ab8b-79f91016dbe1"
version = "1.0.0"

[[deps.ChunkCodecLibZlib]]
deps = ["ChunkCodecCore", "Zlib_jll"]
git-tree-sha1 = "cee8104904c53d39eb94fd06cbe60cb5acde7177"
uuid = "4c0bbee4-addc-4d73-81a0-b6caacae83c8"
version = "1.0.0"

[[deps.ChunkCodecLibZstd]]
deps = ["ChunkCodecCore", "Zstd_jll"]
git-tree-sha1 = "34d9873079e4cb3d0c62926a225136824677073f"
uuid = "55437552-ac27-4d47-9aa3-63184e8fd398"
version = "1.0.0"

[[deps.CloseOpenIntervals]]
deps = ["Static", "StaticArrayInterface"]
git-tree-sha1 = "05ba0d07cd4fd8b7a39541e31a7b0254704ea581"
uuid = "fb6a15b2-703c-40df-9091-08a04967cfa9"
version = "0.1.13"

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

[[deps.CommonWorldInvalidations]]
git-tree-sha1 = "ae52d1c52048455e85a387fbee9be553ec2b68d0"
uuid = "f70d9fcc-98c5-4d4a-abd7-e4cdeebd8ca8"
version = "1.0.0"

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

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.ConstructionBase]]
git-tree-sha1 = "b4b092499347b18a015186eae3042f72267106cb"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.6.0"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.CoordinateTransformations]]
deps = ["LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "a692f5e257d332de1e554e4566a4e5a8a72de2b2"
uuid = "150eb455-5306-5404-9cee-2592286d6298"
version = "0.6.4"

[[deps.CpuId]]
deps = ["Markdown"]
git-tree-sha1 = "fcbb72b032692610bfbdb15018ac16a36cf2e406"
uuid = "adafc99b-e345-5852-983c-f28acb93d879"
version = "0.3.1"

[[deps.CustomUnitRanges]]
git-tree-sha1 = "1a3f97f907e6dd8983b744d2642651bb162a3f7a"
uuid = "dc8bdbbb-1ca9-579f-8c36-e416f6a65cce"
version = "1.0.2"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataStructures]]
deps = ["OrderedCollections"]
git-tree-sha1 = "e357641bb3e0638d353c4b29ea0e40ea644066a6"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.19.3"

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

[[deps.FFTViews]]
deps = ["CustomUnitRanges", "FFTW"]
git-tree-sha1 = "cbdf14d1e8c7c8aacbe8b19862e0179fd08321c2"
uuid = "4f61f5a4-77b1-5117-aa51-3ab5ef4ef0cd"
version = "0.3.2"

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

[[deps.FileIO]]
deps = ["Pkg", "Requires", "UUIDs"]
git-tree-sha1 = "d60eb76f37d7e5a40cc2e7c36974d864b82dc802"
uuid = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
version = "1.17.1"
weakdeps = ["HTTP"]

    [deps.FileIO.extensions]
    HTTPExt = "HTTP"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "273bd1cd30768a2fddfa3fd63bbc746ed7249e5f"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.9.0"

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

[[deps.Giflib_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6570366d757b50fabae9f4315ad74d2e40c0560a"
uuid = "59f7168a-df46-5410-90c8-f2779963d0ec"
version = "5.2.3+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "GettextRuntime_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "50c11ffab2a3d50192a228c313f05b5b5dc5acb2"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.86.0+0"

[[deps.Graphics]]
deps = ["Colors", "LinearAlgebra", "NaNMath"]
git-tree-sha1 = "a641238db938fff9b2f60d08ed9030387daf428c"
uuid = "a2bd30eb-e257-5431-a919-1863eab51364"
version = "1.1.3"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a6dbda1fd736d60cc477d99f2e7a042acfa46e8"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.15+0"

[[deps.Graphs]]
deps = ["ArnoldiMethod", "DataStructures", "Distributed", "Inflate", "LinearAlgebra", "Random", "SharedArrays", "SimpleTraits", "SparseArrays", "Statistics"]
git-tree-sha1 = "7a98c6502f4632dbe9fb1973a4244eaa3324e84d"
uuid = "86223c79-3864-5bf0-83f7-82e725a168b6"
version = "1.13.1"

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

[[deps.HashArrayMappedTries]]
git-tree-sha1 = "2eaa69a7cab70a52b9687c8bf950a5a93ec895ae"
uuid = "076d061b-32b6-4027-95e0-9a2c6f6d7e74"
version = "0.2.0"

[[deps.HistogramThresholding]]
deps = ["ImageBase", "LinearAlgebra", "MappedArrays"]
git-tree-sha1 = "7194dfbb2f8d945abdaf68fa9480a965d6661e69"
uuid = "2c695a8d-9458-5d45-9878-1b8a99cf7853"
version = "0.3.1"

[[deps.HostCPUFeatures]]
deps = ["BitTwiddlingConvenienceFunctions", "IfElse", "Libdl", "Static"]
git-tree-sha1 = "8e070b599339d622e9a081d17230d74a5c473293"
uuid = "3e5b6fbb-0976-4d2c-9146-d79de83f2fb0"
version = "0.1.17"

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

[[deps.IfElse]]
git-tree-sha1 = "debdd00ffef04665ccbb3e150747a77560e8fad1"
uuid = "615f187c-cbe4-4ef1-ba3b-2fcf58d6d173"
version = "0.1.1"

[[deps.ImageAxes]]
deps = ["AxisArrays", "ImageBase", "ImageCore", "Reexport", "SimpleTraits"]
git-tree-sha1 = "e12629406c6c4442539436581041d372d69c55ba"
uuid = "2803e5a7-5153-5ecf-9a86-9b4c37f5f5ac"
version = "0.6.12"

[[deps.ImageBase]]
deps = ["ImageCore", "Reexport"]
git-tree-sha1 = "eb49b82c172811fd2c86759fa0553a2221feb909"
uuid = "c817782e-172a-44cc-b673-b171935fbb9e"
version = "0.1.7"

[[deps.ImageBinarization]]
deps = ["HistogramThresholding", "ImageCore", "LinearAlgebra", "Polynomials", "Reexport", "Statistics"]
git-tree-sha1 = "33485b4e40d1df46c806498c73ea32dc17475c59"
uuid = "cbc4b850-ae4b-5111-9e64-df94c024a13d"
version = "0.3.1"

[[deps.ImageContrastAdjustment]]
deps = ["ImageBase", "ImageCore", "ImageTransformations", "Parameters"]
git-tree-sha1 = "eb3d4365a10e3f3ecb3b115e9d12db131d28a386"
uuid = "f332f351-ec65-5f6a-b3d1-319c6670881a"
version = "0.3.12"

[[deps.ImageCore]]
deps = ["ColorVectorSpace", "Colors", "FixedPointNumbers", "MappedArrays", "MosaicViews", "OffsetArrays", "PaddedViews", "PrecompileTools", "Reexport"]
git-tree-sha1 = "8c193230235bbcee22c8066b0374f63b5683c2d3"
uuid = "a09fc81d-aa75-5fe9-8630-4744c3626534"
version = "0.10.5"

[[deps.ImageCorners]]
deps = ["ImageCore", "ImageFiltering", "PrecompileTools", "StaticArrays", "StatsBase"]
git-tree-sha1 = "24c52de051293745a9bad7d73497708954562b79"
uuid = "89d5987c-236e-4e32-acd0-25bd6bd87b70"
version = "0.1.3"

[[deps.ImageDistances]]
deps = ["Distances", "ImageCore", "ImageMorphology", "LinearAlgebra", "Statistics"]
git-tree-sha1 = "08b0e6354b21ef5dd5e49026028e41831401aca8"
uuid = "51556ac3-7006-55f5-8cb3-34580c88182d"
version = "0.2.17"

[[deps.ImageFiltering]]
deps = ["CatIndices", "ComputationalResources", "DataStructures", "FFTViews", "FFTW", "ImageBase", "ImageCore", "LinearAlgebra", "OffsetArrays", "PrecompileTools", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "TiledIteration"]
git-tree-sha1 = "52116260a234af5f69969c5286e6a5f8dc3feab8"
uuid = "6a3955dd-da59-5b1f-98d4-e7296123deb5"
version = "0.7.12"

[[deps.ImageIO]]
deps = ["FileIO", "IndirectArrays", "JpegTurbo", "LazyModules", "Netpbm", "OpenEXR", "PNGFiles", "QOI", "Sixel", "TiffImages", "UUIDs", "WebP"]
git-tree-sha1 = "696144904b76e1ca433b886b4e7edd067d76cbf7"
uuid = "82e4d734-157c-48bb-816b-45c225c6df19"
version = "0.6.9"

[[deps.ImageMagick]]
deps = ["FileIO", "ImageCore", "ImageMagick_jll", "InteractiveUtils"]
git-tree-sha1 = "8e64ab2f0da7b928c8ae889c514a52741debc1c2"
uuid = "6218d12a-5da1-5696-b52f-db25d2ecc6d1"
version = "1.4.2"

[[deps.ImageMagick_jll]]
deps = ["Artifacts", "Bzip2_jll", "FFTW_jll", "Ghostscript_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "OpenJpeg_jll", "Zlib_jll", "Zstd_jll", "libpng_jll", "libwebp_jll", "libzip_jll"]
git-tree-sha1 = "d670e8e3adf0332f57054955422e85a4aec6d0b0"
uuid = "c73af94c-d91f-53ed-93a7-00f77d67a9d7"
version = "7.1.2005+0"

[[deps.ImageMetadata]]
deps = ["AxisArrays", "ImageAxes", "ImageBase", "ImageCore"]
git-tree-sha1 = "2a81c3897be6fbcde0802a0ebe6796d0562f63ec"
uuid = "bc367c6b-8a6b-528e-b4bd-a4b897500b49"
version = "0.9.10"

[[deps.ImageMorphology]]
deps = ["DataStructures", "ImageCore", "LinearAlgebra", "LoopVectorization", "OffsetArrays", "Requires", "TiledIteration"]
git-tree-sha1 = "895205d762ae24a01689f8cc7ad584b55f1fd005"
uuid = "787d08f9-d448-5407-9aad-5290dd7ab264"
version = "0.4.7"

[[deps.ImageQualityIndexes]]
deps = ["ImageContrastAdjustment", "ImageCore", "ImageDistances", "ImageFiltering", "LazyModules", "OffsetArrays", "PrecompileTools", "Statistics"]
git-tree-sha1 = "783b70725ed326340adf225be4889906c96b8fd1"
uuid = "2996bd0c-7a13-11e9-2da2-2f5ce47296a9"
version = "0.3.7"

[[deps.ImageSegmentation]]
deps = ["Clustering", "DataStructures", "Distances", "Graphs", "ImageCore", "ImageFiltering", "ImageMorphology", "LinearAlgebra", "MetaGraphs", "RegionTrees", "SimpleWeightedGraphs", "StaticArrays", "Statistics"]
git-tree-sha1 = "8071ca812183ee9acb8e93e8d59c66a7d8742d5c"
uuid = "80713f31-8817-5129-9cf8-209ff8fb23e1"
version = "1.10.0"

[[deps.ImageShow]]
deps = ["Base64", "ColorSchemes", "FileIO", "ImageBase", "ImageCore", "OffsetArrays", "StackViews"]
git-tree-sha1 = "3b5344bcdbdc11ad58f3b1956709b5b9345355de"
uuid = "4e3cecfd-b093-5904-9786-8bbb286a6a31"
version = "0.3.8"

[[deps.ImageTransformations]]
deps = ["AxisAlgorithms", "CoordinateTransformations", "ImageBase", "ImageCore", "Interpolations", "OffsetArrays", "Rotations", "StaticArrays"]
git-tree-sha1 = "dfde81fafbe5d6516fb864dc79362c5c6b973c82"
uuid = "02fcd773-0e25-5acc-982a-7f6622650795"
version = "0.10.2"

[[deps.Images]]
deps = ["Base64", "FileIO", "Graphics", "ImageAxes", "ImageBase", "ImageBinarization", "ImageContrastAdjustment", "ImageCore", "ImageCorners", "ImageDistances", "ImageFiltering", "ImageIO", "ImageMagick", "ImageMetadata", "ImageMorphology", "ImageQualityIndexes", "ImageSegmentation", "ImageShow", "ImageTransformations", "IndirectArrays", "IntegralArrays", "Random", "Reexport", "SparseArrays", "StaticArrays", "Statistics", "StatsBase", "TiledIteration"]
git-tree-sha1 = "a49b96fd4a8d1a9a718dfd9cde34c154fc84fcd5"
uuid = "916415d5-f1e6-5110-898d-aaa5f9f070e0"
version = "0.26.2"

[[deps.Imath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "0936ba688c6d201805a83da835b55c61a180db52"
uuid = "905a6f67-0a94-5f89-b386-d35d92009cd1"
version = "3.1.11+0"

[[deps.IndirectArrays]]
git-tree-sha1 = "012e604e1c7458645cb8b436f8fba789a51b257f"
uuid = "9b13fd28-a010-5f03-acff-a1bbcff69959"
version = "1.0.0"

[[deps.Inflate]]
git-tree-sha1 = "d1b1b796e47d94588b3757fe84fbf65a5ec4a80d"
uuid = "d25df0c9-e2be-5dd7-82c8-3ad0b3e990b9"
version = "0.1.5"

[[deps.IntegralArrays]]
deps = ["ColorTypes", "FixedPointNumbers", "IntervalSets"]
git-tree-sha1 = "b842cbff3f44804a84fda409745cc8f04c029a20"
uuid = "1d092043-8f09-5a30-832f-7509e371ab51"
version = "0.1.6"

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

[[deps.IntervalSets]]
git-tree-sha1 = "5fbb102dcb8b1a858111ae81d56682376130517d"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.11"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.IrrationalConstants]]
git-tree-sha1 = "b2d91fe939cae05960e760110b328288867b5758"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.6"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLD2]]
deps = ["ChunkCodecLibZlib", "ChunkCodecLibZstd", "FileIO", "MacroTools", "Mmap", "OrderedCollections", "PrecompileTools", "ScopedValues"]
git-tree-sha1 = "da2e9b4d1abbebdcca0aa68afa0aa272102baad7"
uuid = "033835bb-8acc-5ee8-8aae-3f567f8a3819"
version = "0.6.2"
weakdeps = ["UnPack"]

    [deps.JLD2.extensions]
    UnPackExt = "UnPack"

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

[[deps.JpegTurbo]]
deps = ["CEnum", "FileIO", "ImageCore", "JpegTurbo_jll", "TOML"]
git-tree-sha1 = "9496de8fb52c224a2e3f9ff403947674517317d9"
uuid = "b835a17e-a41a-41e7-81f0-2f016b05efe0"
version = "0.1.6"

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

[[deps.LayoutPointers]]
deps = ["ArrayInterface", "LinearAlgebra", "ManualMemory", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "a9eaadb366f5493a5654e843864c13d8b107548c"
uuid = "10f19ff3-798f-405d-979b-55457f8fc047"
version = "0.1.17"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LazyModules]]
git-tree-sha1 = "a560dd966b386ac9ae60bdd3a3d3a326062d3c3e"
uuid = "8cdb02fc-e678-4876-92c5-9defec4f444e"
version = "0.3.1"

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

[[deps.LittleCMS_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll"]
git-tree-sha1 = "8e6a74641caf3b84800f2ccd55dc7ab83893c10b"
uuid = "d3a379c0-f9a3-5b72-a4c0-6bf4d2e8af0f"
version = "2.17.0+0"

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

[[deps.LoopVectorization]]
deps = ["ArrayInterface", "CPUSummary", "CloseOpenIntervals", "DocStringExtensions", "HostCPUFeatures", "IfElse", "LayoutPointers", "LinearAlgebra", "OffsetArrays", "PolyesterWeave", "PrecompileTools", "SIMDTypes", "SLEEFPirates", "Static", "StaticArrayInterface", "ThreadingUtilities", "UnPack", "VectorizationBase"]
git-tree-sha1 = "a9fc7883eb9b5f04f46efb9a540833d1fad974b3"
uuid = "bdcacae8-1622-11e9-2a5c-532679323890"
version = "0.12.173"

    [deps.LoopVectorization.extensions]
    ForwardDiffExt = ["ChainRulesCore", "ForwardDiff"]
    ForwardDiffNNlibExt = ["ForwardDiff", "NNlib"]
    SpecialFunctionsExt = "SpecialFunctions"

    [deps.LoopVectorization.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    NNlib = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
    SpecialFunctions = "276daf66-3868-5448-9aa4-cd146d93841b"

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

[[deps.ManualMemory]]
git-tree-sha1 = "bcaef4fc7a0cfe2cba636d84cda54b5e4e4ca3cd"
uuid = "d125e4d3-2237-4719-b19c-fa641b8a4667"
version = "0.1.8"

[[deps.MappedArrays]]
git-tree-sha1 = "2dab0221fe2b0f2cb6754eaa743cc266339f527e"
uuid = "dbb5928d-eab1-5f90-85c2-b9b0edb7c900"
version = "0.4.2"

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
git-tree-sha1 = "b513cedd20d9c914783d8ad83d08120702bf2c77"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.3"

[[deps.MetaGraphs]]
deps = ["Graphs", "JLD2", "Random"]
git-tree-sha1 = "3a8f462a180a9d735e340f4e8d5f364d411da3a4"
uuid = "626554b9-1ddb-594c-aa3c-2596fe9399a5"
version = "0.8.1"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MosaicViews]]
deps = ["MappedArrays", "OffsetArrays", "PaddedViews", "StackViews"]
git-tree-sha1 = "7b86a5d4d70a9f5cdf2dacb3cbe6d251d1a61dbe"
uuid = "e94cdb99-869f-56ef-bcf0-1ae2bcbe0389"
version = "0.3.4"

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

[[deps.Netpbm]]
deps = ["FileIO", "ImageCore", "ImageMetadata"]
git-tree-sha1 = "d92b107dbb887293622df7697a2223f9f8176fcd"
uuid = "f09324ee-3d7c-5217-9330-fc30815ba969"
version = "1.1.1"

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

[[deps.OpenEXR]]
deps = ["Colors", "FileIO", "OpenEXR_jll"]
git-tree-sha1 = "97db9e07fe2091882c765380ef58ec553074e9c7"
uuid = "52e1d378-f018-4a11-a4be-720524705ac7"
version = "0.3.3"

[[deps.OpenEXR_jll]]
deps = ["Artifacts", "Imath_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "8292dd5c8a38257111ada2174000a33745b06d4e"
uuid = "18a262bb-aa17-5467-a713-aee519bc75cb"
version = "3.2.4+0"

[[deps.OpenJpeg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libtiff_jll", "LittleCMS_jll", "libpng_jll"]
git-tree-sha1 = "215a6666fee6d6b3a6e75f2cc22cb767e2dd393a"
uuid = "643b3616-a352-519d-856d-80112ee9badc"
version = "2.5.5+0"

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

[[deps.PNGFiles]]
deps = ["Base64", "CEnum", "ImageCore", "IndirectArrays", "OffsetArrays", "libpng_jll"]
git-tree-sha1 = "cf181f0b1e6a18dfeb0ee8acc4a9d1672499626c"
uuid = "f57f5aa1-a3ce-4bc8-8ab9-96f992907883"
version = "0.4.4"

[[deps.PaddedViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "0fac6313486baae819364c52b4f483450a9d793f"
uuid = "5432bcbf-9aad-5242-b902-cca2824c8663"
version = "0.5.12"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1f7f9bbd5f7a2e5a9f7d96e51c9754454ea7f60b"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.56.4+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

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

[[deps.PkgVersion]]
deps = ["Pkg"]
git-tree-sha1 = "f9501cc0430a26bc3d156ae1b5b0c1b47af4d6da"
uuid = "eebad327-c553-4316-9ea0-9fa01ccd7688"
version = "0.3.3"

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

[[deps.PolyesterWeave]]
deps = ["BitTwiddlingConvenienceFunctions", "CPUSummary", "IfElse", "Static", "ThreadingUtilities"]
git-tree-sha1 = "645bed98cd47f72f67316fd42fc47dee771aefcd"
uuid = "1d0040c9-8b98-4ee7-8388-3f51789ca0ad"
version = "0.2.2"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "972089912ba299fba87671b025cd0da74f5f54f7"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.1.0"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieExt = "Makie"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    Makie = "ee78f7c6-11fb-53f2-987a-cfe4a2b5a57a"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.QOI]]
deps = ["ColorTypes", "FileIO", "FixedPointNumbers"]
git-tree-sha1 = "8b3fc30bc0390abdce15f8822c889f669baed73d"
uuid = "4b34888f-f399-49d4-9bb3-47ed5cae4e65"
version = "1.0.1"

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

[[deps.Quaternions]]
deps = ["LinearAlgebra", "Random", "RealDot"]
git-tree-sha1 = "994cc27cdacca10e68feb291673ec3a76aa2fae9"
uuid = "94ee1d12-ae83-5a48-8b1c-48b8ff168ae0"
version = "0.7.6"

[[deps.REPL]]
deps = ["InteractiveUtils", "JuliaSyntaxHighlighting", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.RangeArrays]]
git-tree-sha1 = "b9039e93773ddcfc828f12aadf7115b4b4d225f5"
uuid = "b3c3ace0-ae52-54e7-9d0b-2c1406fd6b9d"
version = "0.3.2"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RealDot]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9f0a1b71baaf7650f4fa8a1d168c7fb6ee41f0c9"
uuid = "c1ae055f-0cd5-4b69-90a6-9a35b1a98df9"
version = "0.1.0"

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

[[deps.RegionTrees]]
deps = ["IterTools", "LinearAlgebra", "StaticArrays"]
git-tree-sha1 = "4618ed0da7a251c7f92e869ae1a19c74a7d2a7f9"
uuid = "dee08c22-ab7f-5625-9660-a9af2021b33f"
version = "0.3.2"

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

[[deps.Rotations]]
deps = ["LinearAlgebra", "Quaternions", "Random", "StaticArrays"]
git-tree-sha1 = "5680a9276685d392c87407df00d57c9924d9f11e"
uuid = "6038ab10-8711-5258-84ad-4b1120ba62dc"
version = "1.7.1"
weakdeps = ["RecipesBase"]

    [deps.Rotations.extensions]
    RotationsRecipesBaseExt = "RecipesBase"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.SIMD]]
deps = ["PrecompileTools"]
git-tree-sha1 = "e24dc23107d426a096d3eae6c165b921e74c18e4"
uuid = "fdea26ae-647d-5447-a871-4b548cad5224"
version = "3.7.2"

[[deps.SIMDTypes]]
git-tree-sha1 = "330289636fb8107c5f32088d2741e9fd7a061a5c"
uuid = "94e857df-77ce-4151-89e5-788b33177be4"
version = "0.1.0"

[[deps.SLEEFPirates]]
deps = ["IfElse", "Static", "VectorizationBase"]
git-tree-sha1 = "456f610ca2fbd1c14f5fcf31c6bfadc55e7d66e0"
uuid = "476501e8-09a2-5ece-8869-fb82de89a1fa"
version = "0.6.43"

[[deps.SciMLPublic]]
git-tree-sha1 = "ed647f161e8b3f2973f24979ec074e8d084f1bee"
uuid = "431bcebd-1456-4ced-9d72-93c2757fff0b"
version = "1.0.0"

[[deps.ScopedValues]]
deps = ["HashArrayMappedTries", "Logging"]
git-tree-sha1 = "c3b2323466378a2ba15bea4b2f73b081e022f473"
uuid = "7e506255-f358-4e82-b7e4-beb19740aa63"
version = "1.5.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "c5391c6ace3bc430ca630251d02ea9687169ca68"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.2"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

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
git-tree-sha1 = "be8eeac05ec97d379347584fa9fe2f5f76795bcb"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.5"

[[deps.SimpleWeightedGraphs]]
deps = ["Graphs", "LinearAlgebra", "Markdown", "SparseArrays"]
git-tree-sha1 = "3e5f165e58b18204aed03158664c4982d691f454"
uuid = "47aef6b3-ad0c-573a-a1e2-d07658019622"
version = "1.5.0"

[[deps.Sixel]]
deps = ["Dates", "FileIO", "ImageCore", "IndirectArrays", "OffsetArrays", "REPL", "libsixel_jll"]
git-tree-sha1 = "0494aed9501e7fb65daba895fb7fd57cc38bc743"
uuid = "45858cf5-a6b0-47a3-bbea-62219f50df47"
version = "0.1.5"

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

[[deps.StackViews]]
deps = ["OffsetArrays"]
git-tree-sha1 = "be1cf4eb0ac528d96f5115b4ed80c26a8d8ae621"
uuid = "cae243ae-269e-4f55-b966-ac2d0dc13c15"
version = "0.1.2"

[[deps.Static]]
deps = ["CommonWorldInvalidations", "IfElse", "PrecompileTools", "SciMLPublic"]
git-tree-sha1 = "49440414711eddc7227724ae6e570c7d5559a086"
uuid = "aedffcd0-7271-4cad-89d0-dc628f76c6d3"
version = "1.3.1"

[[deps.StaticArrayInterface]]
deps = ["ArrayInterface", "Compat", "IfElse", "LinearAlgebra", "PrecompileTools", "Static"]
git-tree-sha1 = "96381d50f1ce85f2663584c8e886a6ca97e60554"
uuid = "0d7ed370-da01-4f52-bd93-41d350b8b718"
version = "1.8.0"
weakdeps = ["OffsetArrays", "StaticArrays"]

    [deps.StaticArrayInterface.extensions]
    StaticArrayInterfaceOffsetArraysExt = "OffsetArrays"
    StaticArrayInterfaceStaticArraysExt = "StaticArrays"

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

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "b117c1fe033a04126780c898e75c7980bf676df3"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.7"

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

[[deps.ThreadingUtilities]]
deps = ["ManualMemory"]
git-tree-sha1 = "d969183d3d244b6c33796b5ed01ab97328f2db85"
uuid = "8290d209-cae3-49c0-8002-c8c24d57dab5"
version = "0.5.5"

[[deps.TiffImages]]
deps = ["ColorTypes", "DataStructures", "DocStringExtensions", "FileIO", "FixedPointNumbers", "IndirectArrays", "Inflate", "Mmap", "OffsetArrays", "PkgVersion", "PrecompileTools", "ProgressMeter", "SIMD", "UUIDs"]
git-tree-sha1 = "98b9352a24cb6a2066f9ababcc6802de9aed8ad8"
uuid = "731e570b-9d59-4bfa-96dc-6df516fadf69"
version = "0.11.6"

[[deps.TiledIteration]]
deps = ["OffsetArrays", "StaticArrayInterface"]
git-tree-sha1 = "1176cc31e867217b06928e2f140c90bd1bc88283"
uuid = "06e1c1a7-607b-532d-9fad-de7d9aa2abac"
version = "0.5.0"

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

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

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

[[deps.VectorizationBase]]
deps = ["ArrayInterface", "CPUSummary", "HostCPUFeatures", "IfElse", "LayoutPointers", "Libdl", "LinearAlgebra", "SIMDTypes", "Static", "StaticArrayInterface"]
git-tree-sha1 = "d1d9a935a26c475ebffd54e9c7ad11627c43ea85"
uuid = "3d5dd08c-fd9d-11e8-17fa-ed2836048c2f"
version = "0.21.72"

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

[[deps.WebP]]
deps = ["CEnum", "ColorTypes", "FileIO", "FixedPointNumbers", "ImageCore", "libwebp_jll"]
git-tree-sha1 = "aa1ca3c47f119fbdae8770c29820e5e6119b83f2"
uuid = "e3aaa7dc-3e4b-44e0-be63-ffb868ccd7c1"
version = "0.1.3"

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

[[deps.libsixel_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "libpng_jll"]
git-tree-sha1 = "c1733e347283df07689d71d61e14be986e49e47a"
uuid = "075b6546-f08a-558a-be8f-8157d0f608a5"
version = "1.10.5+0"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll"]
git-tree-sha1 = "11e1772e7f3cc987e9d3de991dd4f6b2602663a5"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.8+0"

[[deps.libwebp_jll]]
deps = ["Artifacts", "Giflib_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libglvnd_jll", "Libtiff_jll", "libpng_jll"]
git-tree-sha1 = "4e4282c4d846e11dce56d74fa8040130b7a95cb3"
uuid = "c5f90fcd-3b7e-5836-afba-fc50a0988cb2"
version = "1.6.0+0"

[[deps.libzip_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "OpenSSL_jll", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "86addc139bca85fdf9e7741e10977c45785727b7"
uuid = "337d8026-41b4-5cde-a456-74a10e5b31d1"
version = "1.11.3+0"

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
# ╟─9f90a18b-114f-4039-9aaf-f52c77205a49
# ╟─1afdb42f-6bce-4fb1-860a-820d98df0f9d
# ╟─50752620-a604-442c-bf92-992963b1dd7a
# ╟─3e2e1ea8-3a7d-462f-ac38-43a087907a14
# ╟─e31e77ed-5ed6-4689-bfdc-1de73f6fc7b8
# ╟─d61aca4b-a8a7-4876-9310-647c34d3f715
# ╟─bef9bfb4-0521-4cdb-a638-b16c13a60f10
# ╟─7bbf37e1-27fd-4871-bc1d-c9c3ecaac076
# ╟─bc96a33d-9011-41ec-a19e-d472cbaafb70
# ╟─2495aa78-b0dd-4245-b79e-2a6ebf1b9e96
# ╟─c12493fd-862d-4391-b76a-62fdc1948f9d
# ╟─1e927760-4dc6-4ecf-86ec-8fed16dbb0d6
# ╟─ff058cce-d46b-4fa1-83a3-9ed7188f12ae
# ╟─d15e1205-647e-471a-9d71-24e39a7d4801
# ╟─b0ec14d9-0ea0-4c06-a17f-f8bf81c5a1e0
# ╟─3bc38a43-f67b-49c9-a5c9-c7d686e0d675
# ╟─cb80542d-1ad1-402c-9422-18325c554eba
# ╟─9a0e74cb-2f4f-42cb-b7bf-2fed62101851
# ╟─0e1c65f9-e172-4948-a40e-9f681271333e
# ╟─f0bc4e74-ce90-421c-aab2-2c004712e2bd
# ╟─6cc906ec-fb88-4151-9d33-947b5d1510d1
# ╟─0a9985f8-07e4-4ec0-a9eb-3005f95b7e65
# ╟─97c42357-55bd-4805-8df9-cf42c6bf5243
# ╟─0774a6b6-d620-4e02-959f-817a5c13523f
# ╟─aa271ebf-f9ef-453c-8182-0bba6c4a1e45
# ╟─82b15590-8be9-4570-aad9-46f8133236ec
# ╟─382c174d-d6cd-43d1-8c1a-725687f971ee
# ╟─a17437dc-b951-4337-b186-6d55849fe464
# ╟─58624ea1-383d-4a28-b9d2-7a84583ddaa1
# ╟─c8637533-4413-4361-8361-5242b33e4cde
# ╟─6294fe86-2097-403d-bc96-1a4655ef128d
# ╟─a3873cb6-67f1-41bf-b457-b90439868b16
# ╟─d2b9f110-f475-42d4-8996-1db6dc57b91d
# ╟─d34bb337-982b-45b5-ab61-512822a991de
# ╟─e7cd47b9-8fbd-4291-a97e-b75b232dd91c
# ╟─53b602dc-979c-420e-bd42-9147d70730d8
# ╟─681aa669-fb83-4f9d-a025-80b4d9cdc8f7
# ╟─e439723e-5ae5-48dd-bbfe-78a0ff6b8413
# ╟─cc3d2d19-81f8-4f43-be30-81fc2680d1c7
# ╟─0d104033-945d-4c0d-ad1d-1a3750c9adea
# ╟─7d8ed44a-06ba-4432-8345-55bb31eb8f1d
# ╟─9e27864c-f5cb-4780-bcd3-e3d29a69742a
# ╟─5bd15469-e888-4505-a53d-49fef3329ea4
# ╟─3b4a2f77-587b-41fd-af92-17e9411929c8
# ╟─c9e0eaae-b340-434e-bdc9-dfdbc747221e
# ╟─ff61cd9d-a193-44b3-a715-3c372ade7f79
# ╟─443fa256-ee34-43c0-8efd-c12560c00492
# ╟─b742a37d-b49a-467b-bda7-6d39cce33125
# ╟─89df7ccb-53da-4b96-bbb4-fe39109467dd
# ╟─f44b5a95-95b5-4d88-927a-61d91ed51c53
# ╟─221b8f09-6727-4613-8b96-02d70d337280
# ╟─340c9b5b-5ca0-4313-870e-912d5c2dc451
# ╟─bce5c041-be39-4ed1-8935-c389293400bc
# ╟─db6eb97c-558c-4206-a112-6ab3b0ad04c8
# ╟─ae5c476b-193e-44e3-b3a6-36c8944d4816
# ╟─deedb2db-8483-4026-975f-3d5af5a249b7
# ╟─128b6ad8-aa21-4d0a-8124-2cf433bc79c4
# ╟─67f1a8a2-e78a-4220-b127-ae30385822d5
# ╟─403af436-d7f2-43c0-803a-8104ba69fcfd
# ╟─3936846c-52e9-47ab-b557-38081dfdbe12
# ╟─4d99c216-e32f-43a3-a122-ccbb697711fc
# ╟─5b500acf-7029-43ff-9835-a26d8fe05194
# ╟─4d281b64-f9fb-43c3-81de-54b4e7761d6e
# ╟─4bf768de-833f-45bf-9429-4820ff61553f
# ╟─656da51f-fd35-4e89-9af5-b5f0fdf8618f
# ╟─7c03a15f-9ac1-465f-86a4-d2a6087e5970
# ╟─80038fee-b922-479d-9687-771e7e258fcf
# ╟─e28e8089-f52b-440a-9861-895f9c378c84
# ╟─1e52d388-1e8d-4c20-b6e7-bcdd674ea406
# ╟─b662605e-30ef-4e93-a71f-696e76e3ab45
# ╟─59a38e32-c2f3-465f-928d-c05f8d69f496
# ╟─e8fd61f1-33a6-43d8-8056-fb7cf97291b5
# ╟─81ab9972-07bc-4ce9-9138-3359d4e34025
# ╟─a7a24713-a29a-4f0c-996b-f98305bac09c
# ╟─fc9e9bb6-2287-46c8-8518-c9d0804c094e
# ╟─2ce6c56b-733c-42e8-a63b-d774cb6c199c
# ╟─dc8a3e36-2021-42dd-bc49-0eb6ab784fac
# ╟─2e4df75b-0778-4ed4-840a-417da2d65204
# ╟─8730b9a2-a1b4-456c-974c-ecd8880e6834
# ╟─c5be7eb8-e0b6-48cc-8dbe-788fa6624999
# ╟─8a5f2129-4cf5-45d3-8e05-3416b6d8e6fd
# ╟─556617f4-4e88-45f4-9d91-066c24473c44
# ╟─4d6badcc-c061-4e63-a156-167376f131eb
# ╟─c3910dd8-4919-463f-9af0-bc554566c681
# ╟─f664e72d-e762-4dea-ba11-bc8c6b2863f9
# ╟─c134a0b6-8754-48bd-a2ca-932542744407
# ╟─0cffc249-5269-4322-aa4a-ccf7075ec8e1
# ╟─e557ad8b-9e4f-4209-908f-2251e2e2cde9
# ╟─42d137f0-7f7e-4081-ba91-d16c8c8d8d62
# ╟─3bc15a7d-affa-428e-b563-88abb05fe99c
# ╟─2c27858c-79b7-4b58-8cb9-22860c84174b
# ╟─25c1c999-c9b0-437a-99f3-6bb59482ca7d
# ╟─1877ebc8-790f-483a-acf3-9288df9ee7cc
# ╟─0ff13367-204c-43a7-a60b-1daa0df540ac
# ╟─ee656c24-6076-462a-9acb-da66628d61fb
# ╟─39e45305-dd18-477f-a972-b60e8fc9d4ae
# ╟─5df5d3ba-d6e8-4b8d-b7a9-008822a061ba
# ╟─cb4dfdd2-a685-46ea-932f-7df1b1f0bc97
# ╟─64360513-f68f-4ec0-b393-de147a4fa891
# ╟─a198271a-9344-4751-a02f-cf64a80d7147
# ╟─c26ee879-92b6-4d2b-819c-308db5e32ff4
# ╟─271009dc-c42b-40bc-9896-e6fc16238a73
# ╟─a01ebe11-dba2-45df-9fe3-1343576c2071
# ╟─f1274cf1-68b2-4a29-8cb3-0c73f5c1a4fc
# ╟─73b617f0-3339-4d99-80b1-ec01106e3cba
# ╟─75e14d8e-18bc-4fa8-8dcd-b6f5d1c51424
# ╟─67dd55c1-9a5b-4b33-877b-e12192f9c645
# ╟─74c1395f-a049-47ae-b9fe-4ed64e1debd1
# ╟─c2bf647d-542c-47d8-87de-65a644eac516
# ╟─1607c49f-2ece-402a-9abb-6304f96c1dca
# ╟─3b857552-c6aa-44b2-8a56-6c2315bd9a51
# ╟─6004c880-974c-4fb2-897e-b3a0add22508
# ╟─b2490239-a2ba-453e-9a6e-ffb42a1b32b8
# ╟─ddbdd564-750d-4728-b8e2-5f548cddbd05
# ╟─0b1703b4-8a0e-4f3a-9a94-4a778ee451df
# ╟─70b362fd-b3f2-4013-8582-93eddc844029
# ╟─ffd5f874-58ff-4c7c-afa0-6cd4b2dce9af
# ╟─e83c3a60-1e85-4e07-895d-7a3a42cf1f7f
# ╟─361ff521-a117-405f-ba57-91fc2ee00ce2
# ╟─09967f8c-fbc0-4920-9ed6-48e007134026
# ╟─62627b47-5ec9-4d7d-9e94-2148ff198f66
# ╟─fc09b97a-13c9-4721-83ca-f7caa5f55079
# ╟─ef112987-74b4-41fc-842f-ebf1c901b59b
# ╟─95779ca4-b743-43f1-af12-6b14c0e28f0b
# ╟─0f280847-2404-4211-8221-e30418cf4d42
# ╟─b682cc8d-4eeb-4ecd-897c-e15a3e40f76d
# ╟─473ec1bb-4c5e-468b-b19d-335d77868d36
# ╟─6f368af8-84fc-4075-9ee2-ca5740a81c4f
# ╟─4ba80e92-fa56-44f7-b86d-7eafdb70827f
# ╟─0d8509fb-002c-4c4b-b228-f62bebd58c71
# ╟─efc1bfe2-0981-47d9-8cef-7dc8168678ad
# ╟─4b47f07a-7e3e-4769-8398-02776872784c
# ╟─2dd51262-6859-41c3-b0c4-8a17bc19c1de
# ╟─fa805bf5-c3b9-4e5e-b032-9b6961ba12c6
# ╟─acf37074-d5dc-4d7d-96ae-86690c497435
# ╟─d6d93998-1e19-46b6-8a73-8403c1cb5557
# ╟─098a2f02-b0ee-4c05-b023-f5a6303012dd
# ╟─36f9056d-3c77-4f2e-9201-7cd4c2c67ccf
# ╟─1e6a71cf-dc14-45d4-a9c8-7fec8486ffe2
# ╟─7ef9a7d7-3d99-435c-86b5-8de0e0f1dbc1
# ╟─48738f54-f621-4f73-809a-4ae3e4bcedbe
# ╟─3728a7d2-8c06-4a14-b2be-7af880be3040
# ╟─6b9cd068-3faa-4c24-ac14-e30718cee656
# ╟─8081bdc2-2d1f-445b-ace9-b8f0c378f279
# ╟─2302dedd-bafc-45a9-b39f-48c74ed2339d
# ╟─5ef47e4c-3658-4a28-90e6-a30f7ed95ea6
# ╟─0734ddb1-a9a0-4fe1-b5ee-9a839a33d1dc
# ╟─8687dbd1-4857-40e4-b9cb-af469b8563e2
# ╟─fab7a0dd-3a9e-463e-a66b-432a6b2d8a1b
# ╟─4d1ca0d2-9eb0-4166-834c-1641e8bd5a9e
# ╟─61259c61-2c9b-4ead-aa72-b3ecdc21d90b
# ╟─9ac8dd17-0319-46f0-aba4-89b5ce082859
# ╟─63898b4b-b5a5-42bd-aa29-08088208a51b
# ╟─862b0317-f766-47b9-9e57-167f14d534f8
# ╟─4bfb863d-c5f9-4ecb-a946-fde7a2148446
# ╟─279aee95-386a-4c77-81c5-8586400e93b2
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
