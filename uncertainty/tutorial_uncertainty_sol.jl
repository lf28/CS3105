### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ b54db267-1c15-46f5-9b9d-a951a5d9ae0f
using Distributions

# ╔═╡ 72aad7ff-6ed2-4469-8937-93a75b038c0f
using LogExpFunctions  # a very lightweight package that contains xlogx, xlogy, logsumexp functions

# ╔═╡ 9fc16fb3-49c2-47ee-b9a2-1a9b59b0e9e5
using StatsBase

# ╔═╡ 30e4e84b-39dc-4209-b77e-1ae561b173f9
using MCMCChains

# ╔═╡ 0db15737-2193-46a1-8266-070b4d2b7693
using StatsPlots

# ╔═╡ 71d93c0d-0e4a-4684-8a54-6e25c2606a2b
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
	using DataFrames
	
end

# ╔═╡ 428492c0-8ee4-4a48-938b-fa2f4fc93964
TableOfContents()

# ╔═╡ ff753e14-2025-4a94-8faf-d5368f27a8f9
ChooseDisplayMode()

# ╔═╡ 797629b6-0669-11ec-3778-d1f20c3d6a4c
md"""

# CS3105


#### Tutorial: uncertainty
\

$(Resource("https://www.st-andrews.ac.uk/assets/university/brand/logos/standard-vertical-black.png", :width=>130, :align=>"right"))

Lei Fang(@lf28 $(Resource("https://raw.githubusercontent.com/edent/SuperTinyIcons/bed6907f8e4f5cb5bb21299b9070f4d7c51098c0/images/svg/github.svg", :width=>10)))

*School of Computer Science*

*University of St Andrews, UK*

"""

# ╔═╡ 6374b212-ecf4-4887-9e94-fdbb24ca5661
md"""
## Question 1

Given a joint probability distribution over an ensemble of three random variables

| ``A``   | ``B`` | ``C`` | ``P(a,b,c)`` |
|:------:|:------:|:--------:|:----:|
| 1 | 1   | 1     | 0.096 |
| 1 | 1   | 0    | 0.048 |
| 1 | 0 | 1     | 0.064 |
| 1 | 0 | 0    | 0.192 |
| 0 | 1   | 1     | 0.216 |
| 0 | 1   | 0    | 0.048 |
| 0 | 0 | 1     | 0.144 |
| 0 | 0 | 0    | ``p_0`` |

* what is the missing entry ``p_0``'s value?
* use probability rules to evaluate
  * ``P(A)``
  * ``P(A|C=1, B=0)``
  * ``P(A|C=1)``


* are ``A,B`` independent?
"""

# ╔═╡ c7068d15-80be-4a1e-8cbf-f8c62f777b0f
md"""

## What is ``p_0``?
"""

# ╔═╡ 72a517ce-6d49-4c96-b334-52184f98f296

Foldable("Solution", md"""

##### Since ``\sum_{a,b,c} P(a,b,c) =1``, 


$$\large p_0 = P(A=0, B=0, C=0) = 1 - (0.096+\ldots+0.144) =0.192$$

##### _In other words_, we need ``2^3-1 =7`` parameters to specify ``P(A,B,C)``

""")

# ╔═╡ 5be1a24e-f3a5-4672-9b0b-baa116886567
md"""

##

* #### use probability rules to evaluate ``P(A)``


"""

# ╔═╡ 23a7ed25-9c38-4eb2-b4ad-32e0e38c70a8
Foldable("Solution", md"""

$\large P(A=1) = \sum_b\sum_c P(A=1,b,c) = 0.4$


$\large P(A=0) = \sum_b\sum_c P(A=0,b,c) = 0.6$

""")

# ╔═╡ f1d1b12e-3d73-469c-b23d-1534475c07a5
md"""
##

* #### use probability rules to evaluate ``P(A|C=1, B=0)``



"""

# ╔═╡ cfab05b9-09ff-463c-9e52-c86a35e7a6ea
md"""

##
* #### use probability rules to evaluate ``P(A|C=1)``


"""

# ╔═╡ 32e042d4-cbe7-40e4-b479-1896ba38a436
md"""
##
* #### are ``A,B`` independent?


"""

# ╔═╡ 58c2e8d0-dba2-4884-9250-f2aa50af37e1
Foldable("Solution", md"""

#### First, apply sum rule to calculate ``P(A,B)``:

```math
\large P(A,B) = \sum_c P(A,B,c)
```

For example, 

$$\large\begin{align}
P(A=1, B=1) &= P(A=1, B=1, C=1) + P(A=1, B=1, C=0) \\ 
&= 0.096+0.048\\
&=0.144
\end{align};$$

#### The marginal joint ``P(A,B)`` is

$$\large \begin{array}{c|cc} 
P(A, B) & B=1 & B=0 \\
\hline
A=1 & 0.144 & 0.256 \\ 
A=0 & 0.264 & 0.336 \\ \end{array}$$


```math
\large P(A) = \sum_b P(A, b) = \begin{cases}0.144+ 0.256 =0.4 & A=1 \\ 0.264+0.336=0.6 & A=0\end{cases}

```


```math
\large P(B) = \sum_a P(a, B) = \begin{cases}0.408 & B=1 \\ 0.592 & B=0\end{cases}
```

#### Are ``A,B`` independent? **NO**, since

```math
\large \underbrace{P(A =1, B=1)}_{0.144} \neq \underbrace{P(A=1)}_{0.4} \underbrace{P(B=1)}_{0.408}
```
""")

# ╔═╡ b999fe4b-8c22-4844-a9b7-15d7cc0052f2
begin
	Ps_ = [0.096, 0.048, 0.064, 0.192, 0.216, 0.048, 0.144]
	Ps = push!(Ps_, 1-sum(Ps_))
	Pdist = DataFrame(A = [ones(Bool, 4); zeros(Bool, 4)], B= [x for _ in 1:2 for x in [true, false] for _ in 1:2], C= [x for _ in 1:2 for _ in 1:2 for x in [true, false] ], P = Ps)
end;

# ╔═╡ 6414fac6-de38-4a71-aacc-76c6df248225
Foldable("Solution", md"""




```math
\Large P(A|C=1, B=0) \propto P(A, B=0, C=1)
```

* ##### just find the matching rows then normalise

$(Pdist[(Pdist.B .== false) .& (Pdist.C .== true), :])

```math
\large P(A|C=1, B=0) \propto \begin{cases}P(A=1, B=0, C=1)= 0.064 & A=1 \\ P(A=0, B=0, C=1)= 0.144 & A=0\end{cases}

```

```math

\large P(A|C=1, B=0) = \begin{cases}0.308 & A=1 \\ 0.692 & A=0\end{cases}

```
""")

# ╔═╡ bf204cbd-386f-4456-b1d4-38106826a56d
Foldable("Solution", md"""

#### Alternatively, like `enum-ask`

$$\large \begin{align}P(A|C=1) &\propto P(A, C=1) \\
&= \sum_b P(A, b, C=1)\\


&= \begin{cases}0.216+0.144 = 0.36  & A=0 \\  0.096+0.064 = 0.16& A=1\end{cases} 

\end{align}$$

* #### then normalise


$$\large \begin{align}P(A|C=1) 
&= \begin{cases}\frac{36}{36+16}  & A=0 \\ \frac{16}{36+16} & A=1\end{cases} 

\end{align}$$



$(Pdist[(Pdist.C .== true), :])

\



""")

# ╔═╡ 6658512a-f80f-4ef7-b927-2fee89e670e3
TwoColumn(Pdist,  begin
	Pac = Pdist[(Pdist.B .== true), Not(:B)]
	Pac.P = Pac.P + Pdist[(Pdist.B .== false), Not(:B)].P
	Pac
end);

# ╔═╡ afe11239-08b6-4053-88e4-8795dc62ac04
Foldable("Solution", md"""

#### We can first find $P(A,C)$ 

$$\Large P(A, C) = \sum_b P(A, C)$$

$(Pac)



#### then select + normalise


$$\large \begin{align}P(A|C=1) &\propto \begin{cases} 0.36& A= 0\\ 0.16 & A=1\end{cases} \\

&= \begin{cases} \frac{36}{36+16}& A= 0\\  \frac{16}{36+16} & A=1\end{cases}\end{align}$$





""")

# ╔═╡ 283871df-944a-4223-a7e1-043080542843
md"""
## Question 2


(**Conditional independence definitions**) Show that if $X, Y$ are conditionally independent given $Z$, *i.e.* 

$P(X,Y|Z) = P(X|Z)P(Y|Z),$ 
then 

$P(X|Z,Y)= P(X|Z),\;\; P(Y|Z,X) = P(Y|Z).$
"""

# ╔═╡ 8cfe4da4-4cd9-4d75-9446-b5bc1cc122fd
Foldable("Solution", md"""

### Proof ``P(X,Y|Z) = 	P(X|Z)P(Y|Z) \Leftrightarrow P(X| Y, Z)= P(X|Z)``

* ##### from the left hand side, we have
```math
\Large P(X, Y|Z) = P(X|Z)P(Y|Z)
```

* ##### multiply ``P(Z)`` on both sides, we have

```math
\Large \begin{array}{c}
P(X, Y|Z)P(Z) = P(X|Z) P(Y|Z)P(Z)\\
\Updownarrow \\
P(X, Y, Z) = P(X|Z) P(Y, Z) \\
\Updownarrow \\

\frac{P(X, Y, Z)}{P(Y, Z)} = P(X|Z) \\

\Updownarrow \\

P(X| Y, Z)= P(X|Z)
\end{array}
```


""")

# ╔═╡ 0db3a069-725a-4d40-a1b7-e16aa2fbc5f2
# aside(tip(md"""

# Actually, the following three are **equivalent**:


# ```math
# P(X, Y) = P(X)P(Y);
# ```

# ```math
# P(X|Y)=P(X)\; \text{or}\; P(Y|X)=P(Y);
# ```


# ```math
# \begin{align}
# \forall y_1, & y_2 \in \Omega_Y, \\
# & P(X|Y=y_1)=P(X|Y=y_2) \\

# \end{align}
# ```


# Similarly identities hold for conditional independence.
# """))

# ╔═╡ 6009b022-8bc6-443b-87b0-6f75db590da2
# aside(tip(md"""
# $P(X\textcolor{red}{|Z}, Y) = \frac{P(X\textcolor{red}{|Z})P(Y\textcolor{red}{|Z}, X)}{P(Y\textcolor{red}{|Z})}$ is the _conditional version of Bayes' rule_.

# That is: we add the condition "``\;|Z\;``" to each term of the ordinary Bayes' rule

# ```math
# P(X|Y) = \frac{P(X)P(Y|X)}{P(Y)}
# ```


# """))

# ╔═╡ 93a14c73-5039-4099-9068-60c25b20accd
# begin
# 	using TikzGraphs
# 	using Graphs
# end

# ╔═╡ a584b032-e693-48c6-8242-f408187168c6
md"""
## Question 3
Consider a dice rolling scenario: two fair dice are rolled independently (denote the rolling results as ``D_1``, ``D_2``), and you are told the sum of the two rolls. That is if denote the sum as $S$, we have  ``S=D_1+D_2``.  



1. What are $P(D_1=3|S=9, D_2 = 6)$ and $P(D_1=3|S=9, D_2 = 4)$ ? 


2. Construct a BN for the problem


3. Use enum-ask to compute 
    * ``P(D_1 = 3|S=9)``


4. Now instead of being told ``S``, you are only told whether ``S`` is even or not, denote the random variable as $E$, modify your network accordingly
"""

# ╔═╡ 61f6e84a-25dc-4def-9dc3-5faa1392dfa7
md"""
##
#### What are $P(D_1=3|S=9, D_2 = 6)$ and $P(D_1=3|S=9, D_2 = 4)$ ? 



"""

# ╔═╡ 1a944206-1c2e-47c7-8083-e28e9f60bd7d
Foldable("Solution", md"""

#### We don't need to compute, just use common sense
$\Large P(D_1=3|S=9, D_2 = 6) = 1$ 


$\Large P(D_1=3|S=9, D_2 = 4) = 0$  

""")

# ╔═╡ d5039ee8-2e0a-4b66-aae3-355d3d7f40d3
md"""
##
#### Construct a BN for the problem

"""

# ╔═╡ b12ae2b7-01e1-4dee-b217-259651a360e2
Foldable("Solution", md"""


$(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/
catan.svg' width = '400' /></center>")



$$\large P(D_1) = \begin{cases}\frac{1}{6} & D_1=1 \\ \frac{1}{6} & D_1=2  \\ \vdots & \vdots \\ \frac{1}{6} & D_1= 6\end{cases};\;\;\;\; P(D_2) = \begin{cases}\frac{1}{6} & D_2=1 \\ \frac{1}{6} & D_2=2  \\ \vdots & \vdots \\ \frac{1}{6} & D_2= 6\end{cases}$$

\


$$\large P(S|D_1, D_2) = \begin{cases}1 &  S=D_1+D_2 \\ 0 & \text{other wise}\end{cases}= I(S =D_1+D_2)$$
""")

# ╔═╡ e9778657-476e-4920-9ffe-596dc283da80
md"""
##
#### Compute ``P(D_1=3|S=9)`` by `enum-ask`

"""

# ╔═╡ 8f5c045c-3191-4031-92fd-c8975b0e3a35
Foldable("Exact inference: enum-ask", md"""

#### `enum-ask`
$$\Large \begin{align}P(D_1|S=9) &\propto P(D_1, S=9) \\ 
&= \sum_{d_2} P(D_1, D_2=d_2, S=9)\\

&=  \sum_{d_2} P(D_1)P(D_2=d_2)P(S=9|D_1, d_2) \\
&= P(D_1)\sum_{d_2 =1, \ldots, 6} P(D_2=d_2)P(S=9|D_1, d_2) \\
\end{align}$$


""")

# ╔═╡ bacfddfa-bf66-4ad1-8c7d-f53df38ccd67
Foldable("D1=1", md"""


##### For ``D_1=1``, we have 
$$\begin{align}P(D_1=1|S=9) 
&\propto P(D_1=1)\sum_{d_2 =1, \ldots, 6} P(D_2=d_2)P(S=9|D_1=1, d_2) \\
&= \frac{1}{6}\sum_{d_2 =1, \ldots, 6} \frac{1}{6}\times I(9 = 1+d_2)\\
&=\frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  I(9 = 1+1)& d_1=1,d_2 =1 \\ \frac{1}{6} \times  I(9 = 1+2)& d_1=1,d_2 =2 \\ \frac{1}{6} \times  I(9 = 1+3)& d_1=1,d_2 =3 \\ \frac{1}{6} \times  I(9 = 1+4)& d_1=1,d_2 =4 \\ \frac{1}{6} \times  I(9 = 1+5)& d_1=1,d_2 =5 \\ \frac{1}{6} \times  I(9 = 1+6)& d_1=1,d_2 =6  \end{cases} \\

&= \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1=1,d_2 =1 \\ \frac{1}{6} \times  0& d_1=1,d_2 =2 \\ \frac{1}{6} \times  0& d_1=1,d_2 =3 \\ \frac{1}{6} \times 0 & d_1=1,d_2 =4 \\ \frac{1}{6} \times  0& d_1=1,d_2 =5 \\ \frac{1}{6} \times  0& d_1=1,d_2 =6  \end{cases} \\

&= 0
\end{align}$$

""")

# ╔═╡ 3e24cb61-fbc6-4e68-8d51-10447544b1ad
Foldable("D1=2", md"""


##### For ``D_1=2``, we have 
$$\begin{align}P(D_1=2|S=9) 
&\propto P(D_1=2)\sum_{d_2 =1, \ldots, 6} P(D_2=d_2)P(S=9|D_1=2, d_2) \\
&= \frac{1}{6}\sum_{d_2 =1, \ldots, 6} \frac{1}{6}\times I(9 = 2+d_2)\\
&=\frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  I(9 = 2+1)& d_1=2, d_2 =1 \\ \frac{1}{6} \times  I(9 = 2+2)& d_1=2,d_2 =2 \\ \frac{1}{6} \times  I(9 = 2+3)& d_1=2,d_2 =3 \\ \frac{1}{6} \times  I(9 = 2+4)& d_1=2,d_2 =4 \\ \frac{1}{6} \times  I(9 = 2+5)& d_1=2,d_2 =5 \\ \frac{1}{6} \times  I(9 = 2+6)& d_1=2,d_2 =6  \end{cases} \\

&= \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1=2,d_2 =1 \\ \frac{1}{6} \times  0& d_1=2,d_2 =2 \\ \frac{1}{6} \times  0& d_1=2,d_2 =3 \\ \frac{1}{6} \times 0 & d_1=2,d_2 =4 \\ \frac{1}{6} \times  0& d_1=2,d_2 =5 \\ \frac{1}{6} \times  0& d_1=2,d_2 =6  \end{cases}\;\;= 0
\end{align}$$
""")

# ╔═╡ ca522196-9709-4e55-b7d7-62c90f679453
Foldable("D1=3", md"""


##### For ``D_1=3``, we have 
$$\begin{align}P(D_1=3|S=9) 
&\propto \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1=3, d_2 =1 \\ \frac{1}{6} \times  0& d_1=3, d_2 =2 \\ \frac{1}{6} \times  0& d_1=3, d_2 =3 \\ \frac{1}{6} \times 0 & d_1=3, d_2 =4 \\ \frac{1}{6} \times  0& d_1=3, d_2 =5 \\ \frac{1}{6} \times  1& d_1=3, d_2 =6  \end{cases}\;\;= \frac{1}{36}
\end{align}$$
""")

# ╔═╡ d432f301-c487-4b47-aebb-72b9bff7c7b1
Foldable("D1=4", md"""


##### For ``D_1=4``, we have 
$$\begin{align}P(D_1=4|S=9) 
&\propto  \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1=4, d_2 =1 \\ \frac{1}{6} \times  0& d_1=4, d_2 =2 \\ \frac{1}{6} \times  0& d_1=4, d_2 =3 \\ \frac{1}{6} \times 0 & d_1=4,  d_2 =4 \\ \frac{1}{6} \times  1& d_1=4, d_2 =5 \\ \frac{1}{6} \times  0& d_1=4, d_2 =6  \end{cases}\;\;= \frac{1}{36}
\end{align}$$
""")

# ╔═╡ 361d1d0e-949d-40a5-afbd-a32a052d426e
Foldable("D1=5", md"""


##### For ``D_1=5``, we have 
$$\begin{align}P(D_1=5|S=9) 
&= \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1= 5,d_2 =1 \\ \frac{1}{6} \times  0& d_1= 5,d_2 =2 \\ \frac{1}{6} \times  0& d_1= 5,d_2 =3 \\ \frac{1}{6} \times 1 & d_1= 5,d_2 =4 \\ \frac{1}{6} \times  0& d_1= 5,d_2 =5 \\ \frac{1}{6} \times  0& d_1= 5,d_2 =6  \end{cases}\;\;= \frac{1}{36}
\end{align}$$
""")

# ╔═╡ 966e5687-af01-4b2d-afd3-5a2929420d94
Foldable("D1=6", md"""


##### For ``D_1=6``, we have 
$$\begin{align}P(D_1=6|S=9) 
&\propto \frac{1}{6}\times (+)\begin{cases}\frac{1}{6} \times  0& d_1= 6,d_2 =1 \\ \frac{1}{6} \times  0& d_1= 6,d_2 =2 \\ \frac{1}{6} \times  1& d_1= 6,d_2 =3 \\ \frac{1}{6} \times 0 & d_1= 6,d_2 =4 \\ \frac{1}{6} \times  0& d_1= 6,d_2 =5 \\ \frac{1}{6} \times  0& d_1= 6,d_2 =6  \end{cases}\;\;= \frac{1}{36}
\end{align}$$

##### For ``D_1=7``? No need. $P(D_1=7)=0$

##### Therefore, we have 

$$\large \begin{align}P(D_1|S=9) &\propto \left \langle 0, 0, \frac{1}{36},   \frac{1}{36}, \frac{1}{36}, \frac{1}{36}\right \rangle \\ 
&= \left \langle 0, 0, \frac{1}{4},   \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\right \rangle \\ 
&= \begin{cases} 0 & D_1=1 \\ 0 & D_1=2 \\ \frac{1}{4} & D_1=3 \\ \frac{1}{4} & D_1=4\\ \frac{1}{4} & D_1=5\\ \frac{1}{4} & D_1=6\end{cases}
\end{align}$$
""")

# ╔═╡ 49f7774e-14e0-49fb-985c-fa7f5f07f6c8
Foldable("Finally, normalise", md"""
##### Therefore, we have 

$$\large \begin{align}P(D_1|S=9) &\propto \left \langle 0, 0, \frac{1}{36},   \frac{1}{36}, \frac{1}{36}, \frac{1}{36}\right \rangle \\ 
&= \left \langle 0, 0, \frac{1}{4},   \frac{1}{4}, \frac{1}{4}, \frac{1}{4}\right \rangle \\ 
&= \begin{cases} 0 & D_1=1 \\ 0 & D_1=2 \\ \frac{1}{4} & D_1=3 \\ \frac{1}{4} & D_1=4\\ \frac{1}{4} & D_1=5\\ \frac{1}{4} & D_1=6\end{cases}
\end{align}$$

#### Therefore, $P(D_1=3|S=9) =\frac{1}{4}$
""")

# ╔═╡ 1b0f0a64-8868-41af-b609-028ae3090780
md"""
##
#### Rejection sampling to compute $P(D_1=3|S=9)$


"""

# ╔═╡ 9f0ed17d-0f14-47cf-ac89-b6803a7174a1
Foldable("Rejection sampling", md"""

##### Find a topological order: ``D_1, D_2, S``
----
#### for ``m = 1,\ldots, M``
* ##### sample $d_1^{(m)} \sim P(D_1)$
* ##### sample $d_2^{(m)} \sim P(D_2)$
* ##### sample $s^{(m)} \sim P(S|d_1^{(m)}, d_2^{(m)})$
* ##### store $(d_1^{(m)}, d_2^{(m)}, s^{(m)})$
----

##### Reject those inconsistent samples where $s^{(m)} \neq 9$

##### Count the frequency of the remaining samples 
$$\large P(D_1=3|S=9) \approx \frac{1}{M'}\sum_{m'} {I}(d_1^{(m')} = 3)$$
""")

# ╔═╡ 9cb4bec9-b655-433c-8ebc-237a5fbbd0e9
m_sample_size = 5000

# ╔═╡ 542856d2-41df-49e2-9ee1-d02968a93f5e
samples = let
	Random.seed!(2345)
	samples = []
	for m in 1:m_sample_size
		d1 = rand(1:6)
		d2 = rand(1:6)
		s = d1+ d2
		# rejection 
		if s == 9
			push!(samples, d1)
		end
	end

	# samples = hcat(samples...)
	mean(samples .== 3)
end

# ╔═╡ f7e4846c-be4c-43ad-905c-6662afd30981
let
	Random.seed!(1234)
	# M = 5000
	d1_samples = rand(1:6, m_sample_size)
	d2_samples = rand(1:6, m_sample_size)
	s_samples = d1_samples + d2_samples
	mean(d1_samples[s_samples .== 9] .== 3)
end

# ╔═╡ 5671cf5c-e5a8-45eb-8bb2-b021ff98409e
md"""
##
##### Update the BN with S is even or not
"""

# ╔═╡ b64720e2-8be7-42e9-a7ce-4a049520ac25
Foldable("Solution", md"""


$(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/
cataneven.svg' width = '400' /></center>")



$$P(D_1) = \begin{cases}\frac{1}{6} & D_1=1 \\ \frac{1}{6} & D_1=2  \\ \vdots & \vdots \\ \frac{1}{6} & D_1= 6\end{cases};\;\;\;\; P(D_2) = \begin{cases}\frac{1}{6} & D_2=1 \\ \frac{1}{6} & D_2=2  \\ \vdots & \vdots \\ \frac{1}{6} & D_2= 6\end{cases}$$



$$P(S|D_1, D_2) = \begin{cases}1 &  S=D_1+D_2 \\ 0 & \text{other wise}\end{cases}= I(S =D_1+D_2)$$


$$\begin{align}P(E|S) &= \begin{cases} 1& \text{S is even}, E=true\\0& \text{S is even}, E=false\\0 &  \text{S is odd}, E=true\\ 1 &  \text{S is odd}, E=false\end{cases}\end{align}$$


or $$P(E|S) = (1^{I(E=true)}0^{I(E=false)})^{I(\text{iseven}(S))}(1^{I(E=false)}0^{I(E=true)})^{I(\text{isodd}(S))}$$

or $$P(E=true|S)= I(\text{iseven}(S))$$


""")

# ╔═╡ 258d0224-650e-44e5-a28c-43ab141c5053
# md"""

# ##### Gibbs sampling


# $$\large P(D_1=1|E=true)=?$$


# #### Gibbs sampling iteratively samples the non-evidence nodes are $D_1, D_2, S$ based on their full conditionals

# ----
# ##### Initialisation:
# * ##### randomly guess $d_1^{(0)}, d_2^{(0)}, s^{(0)}$


# ##### for ``m = 1:M``
# * ##### sample ``d_1^{(m)}\sim P(D_1|d_2^{(m-1)}, s^{(m-1)}, E=true)``
# * ##### sample ``d_2^{(m)}\sim P(D_2|d_1^{(m)}, s^{(m-1)}, E=true)``
# * ##### sample ``s^{(m)}\sim P(S|d_2^{(m)}, d_1^{(m)}, E=true)``

# * ##### collect ``(d_1^{(m)}, d_2^{(m)}, s^{(m)})``

# ----

# ##### Lastly, do frequency counting 

# $$P(D_1=1|E=true) \approx \frac{1}{M}\sum_{m}^M I(d_1^{(m)}=1)$$


# ##### Full conditionals

# $P(D_1|d_2, s, E=true) \propto 	P(D_1, d_2, s, +e) = P(D_1)\cancel{P(d_2)}P(s|D_1, d_2)\cancel{P(+e|s)}$


# $P(D_2|d_1, s, E=true) \propto 	P(d_1, D_2, s, +e) = \cancel{P(d_1)}{P(D_2)}P(s|D_1, d_2)\cancel{P(+e|s)}$



# $P(S|d_1,d_2, +e) \propto 	P(d_1, d_2, S, +e) = \cancel{P(d_1)}\cancel{P(d_2)}P(S|d_1, d_2){P(+e|S)}$
# """

# ╔═╡ 6582f9c8-9161-4dec-9401-24487b26c0f3
md"""


## Question 4 (Coin switch problem)

> Your friend has two coins: one fair (i.e. with the probability of a head turning up  $p_1=0.5$) and the other is bent with the probability of a head turning up $p_2= 0.2$.
>
> He always uses the fair coin at the beginning but switches to the bent coin at some unknown time.
>
> You have observed the following tossing results: 
> * ``\mathcal{D}= [t,h,h,t,h,t,t]``. 
> And you want to know when he switched the coin.


* Develop a Bayes network for the problem; 


!!! hint "Hint"
	We have already solved the problem in lecture 2 and we have used the conditional independence assumption below

	```math
	P(\mathcal{D}|S) =\prod_{t=1}^7 P(Y_t|S),
	```

	where ``S`` is the unknown switch point and ``Y_t`` is the observation for ``t=1\ldots 7``. We have also used un-informative prior for ``P(S)``. Therefore, you only need to give the graph and also the CPFs based on the generative model.




* Now assume we **donot know** the two coins' biases. Create a Bayes net for the new problem.


!!! hint "Hint"
	Do not start over again, extend your existing Bayes' network instead. 
"""

# ╔═╡ d469deda-38d5-43bb-b638-3b058c046060
md"""

##

#### Bayes' network for the switching problem given the biases
"""

# ╔═╡ fd78a28e-ff0c-4c94-be1e-8870b1b36060
Foldable("Solution", md"""

#### The random variables are


* ##### hypothesis: ``S\in \{1,2,\ldots, N-1\}``, the unknown switching point
  * ##### *after* ``S``, he makes the switch
* ##### data (observation): ``\{Y_1, Y_2,\ldots, Y_N\}``


$(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/
changepoint0.svg' width = '600' /></center>")

#### *CPTs are*

* ##### ``P(S)``: a uniform prior on the unknown switching point:

$$\large P(S) = \begin{cases} \frac{1}{6} & S \in \{1,2,\ldots, 6\} \\ 0 & \text{otherwise}\end{cases}$$



* ##### ``P(Y_i|S)``: for ``i=1\,\ldots, 7``


$$\large P(Y_i|S) = \begin{cases} 0.5& Y_i=1, i\leq S\\ 0.5 &Y_i=0, i\leq S \\
0.2& Y_i=1, i> S\\ 0.8 &Y_i=0, i> S \\ 0& \text{all others}\end{cases}$$



* ##### You may want to write the CPF in one-line: for $$Y_i \in \{0,1\}$$ and $$S = \{1,2,\ldots, N-1\}$$:

$$\large P(Y_i|S) = \left(p_1^{Y_i}(1-p_1)^{1-Y_i}\right)^{I(i\leq S)}\left(p_2^{Y_i}(1-p_2)^{1-Y_i}\right)^{I(i>S)}.$$
  * note that ``p_1=0.5, p_2=0.2``
""")

# ╔═╡ 502598df-9e7a-4887-aa76-ee52368f9bcd
md"""

##
### What if we do not know ``p_1, p_2``?

"""

# ╔═╡ 564dfbdb-ebdd-432c-b3c9-29d429360c28
Foldable("Solution", md"""

#### The random variables become

  * ##### ``S``: switching points and $S \in \{1,2,3,\ldots N-1\}$


  * ##### $P_1$, $P_2$: the unknown probabilities of the two coins


  * ##### coin tossing data $\mathcal{D} = \{Y_1,Y_2, \ldots, Y_N\}$; $Y_i$ is either 1 or 0; 


$(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/changepoint1.svg' width = '600' /></center>")


""")

# ╔═╡ a4b5e04b-2be1-4ad8-8025-94957e98623a
Foldable("The CPTs", md"""

#### **CPTs**


#### ``P(P_1)``: uniform prior 

$P(P_1) = \begin{cases} \frac{1}{11} & P_1 \in \{0, 0.1, \ldots, 1.0\} \\ 0 & \text{otherwise}\end{cases}$

#### ``P(P_2)``: uniform prior 

$P(P_2) = \begin{cases} \frac{1}{11} & P_2 \in \{0, 0.1, \ldots, 1.0\} \\ 0 & \text{otherwise}\end{cases}$

#### ``P(S)``: uniform prior on the unknown switching point:

$P(S) = \begin{cases} \frac{1}{N-1} & S \in \{1,2,\ldots, N-1\} \\ 0 & \text{otherwise}\end{cases}$



#### ``P(Y_i|S, P_1, P_2)``: 


$\large P(Y_i|S, P_1, P_2) = \begin{cases} P_1& Y_i=1, i\leq S\\ 1-P_1 &Y_i=0, i\leq S \\
P_2& Y_i=1, i> S\\ 1-P_2 &Y_i=0, i> S \\ 0& \text{all others}\end{cases}$

""")

# ╔═╡ 18f00394-153f-4260-b5ec-3c26aad32eb9
md"""

##
#### Gibbs sampling 

"""

# ╔═╡ d086fce9-74ea-4f22-9ec9-84c12aca7867
Foldable("Gibbs sampling", md"""

##### _Gibbs sampling_ samples the non-evidenced nodes interatively based on their full conditionals

* ##### the non-evidenced nodes: ``S, P_1, P_2`` 
* ##### where the evidence are fixed to the observed ``y_1, y_2, \ldots, y_7``



$(html"<center><img src='https://leo.host.cs.st-andrews.ac.uk/figs/gibbsswitch.svg' width = '600' /></center>")



\

#### The algorithm:

--------------------------

##### Step 0. initialise random guess ``s^{(0)}, p_1^{(0)}, p_2^{(0)}``

##### for ``i = 1:M``
*  ##### sample ``s^{(i)} \sim P(S| \mathcal{D}=\mathbf{y}, p_1^{(i-1)}, p_2^{(i-1)})``
*  ##### sample ``p_1^{(i)} \sim P(P_1| \mathcal{D}=\mathbf{y}, s^{(i)}, p_2^{(i-1)})``
*  ##### sample ``p_2^{(i)} \sim P(P_2| \mathcal{D}=\mathbf{y}, s^{(i)}, p_1^{(i)})``
*  ##### store sample ``[s^{(i)}, p_1^{(i)}, p_2^{(i)}]``

--------------------------


###### Lastly, do frequency counting 

$$P(S=\lfloor N/2 \rfloor |\mathcal{D}=\mathbf{y}) \approx \frac{1}{M}\sum_{m} I(s^{(m)} = \lfloor N/2 \rfloor )$$
""")

# ╔═╡ ef2bcd80-8975-42d7-a7da-15045d575047
Foldable("Full conditionals", md"""

#### The full conditionals

$$\large P(S|\mathcal{D}=\mathbf{y}, p_1, p_2) \propto \cancel{P(p_1)P(p_2)}P(S)\prod_{i=1}^n P(y_i|S, p_1, p_2)$$


$$\large \begin{align}P(p_1|\mathcal{D}=\mathbf{y}, s, p_2) &\propto {P(P_1)}\cancel{P(p_2)}\cancel{P(s)}\prod_{i=1}^n P(y_i|s, P_1, p_2)
\end{align}$$




$$\large \begin{align}P(p_2|\mathcal{D}=\mathbf{y}, s, p_1) &\propto \cancel{P(p_1)}{P(P_2)}\cancel{P(s)}\prod_{i=1}^n P(y_i|s, p_1, P_2)
\end{align}$$


""")

# ╔═╡ a66da406-fc81-47bc-8745-db1959340ad9
md"
##

#### What Gibbs sampling is doing essentially?"

# ╔═╡ 3e2a88ec-138b-4ec5-a87d-bb2a876fd308
Foldable("Interpretation of Gibbs sampling", md"""

#### Each full conditional cooresponds to a  local inference 

#### Gibbs sampling solves the global problem by reducing it to some smaller ones and "solve" them iteratively

* ##### knowing the biases $P_1=p_1, P_2=p_2$, when did he switch?

$$\large P(S|\mathcal{D}=\mathbf{y}, p_1, p_2) \propto \cancel{P(p_1)P(p_2)}P(S)\prod_{i=1}^n P(y_i|S, p_1, p_2)$$



* ##### knowing the switch point $S=s$, what is the bias of the first coin (i.e. before the switch)

$$\large \begin{align}P(p_1|\mathcal{D}=\mathbf{y}, s, p_2) &\propto {P(P_1)}\cancel{P(p_2)}\cancel{P(s)}\prod_{i=1}^n P(y_i|s, P_1, p_2) \\ 
&= {P(P_1)} \underbrace{\prod_{i=1}^{s} P(y_i| P_1)}_{\text{first coin's tosses}} \cancel{\prod_{i=s+1}^{N} P(y_i|p_2)}
\end{align}$$

* ##### knowing the switch point $S=s$, what is the bias of the second coin (i.e. after the switch)

$$\large \begin{align}P(p_2|\mathcal{D}=\mathbf{y}, s, p_1) &\propto \cancel{P(p_1)}{P(P_2)}\cancel{P(s)}\prod_{i=1}^n P(y_i|s, p_1, P_2) \\ 
&= {P(P_2)} \cancel{\prod_{i=1}^{s} P(y_i| p_1)}\underbrace{\prod_{i=s+1}^{N} P(y_i|P_2)}_{\text{second coin's tosses}} 
\end{align}$$


""")

# ╔═╡ 818ade09-6830-45d5-ae0c-ed73c76c656c
md"""
##
### Demonstration
"""

# ╔═╡ c11b8f03-b0ba-41c5-8b6d-8b2e6fb4489c
begin
	Random.seed!(2345)
	T = 500
	thred = 50
# 	random switch in the middle range of tosses
	truet0 = Int(floor(T/2))
	# truet0 = 270
# 	a relatively fair coin
	p₁ = 0.5
# 	a bent second coin
	p₂ = 0.2
# 	Data: an array of Boolean results 
	D = Vector{Bool}(undef, T)
# 	fill the data before switch; i.e. random draw samples from the first coin tossing
	D[1:truet0] = rand(Bernoulli(p₁), truet0)
# 	random draw samples from the second coin
	D[(truet0+1):end] = rand(Bernoulli(p₂), T-truet0)
# 	scatter((1:truet0)[D[1:truet0] .== 1], D[1:truet0][D[1:truet0] .== 1], label=L"C_1"*" Tosses", m=:circle, c=1,ms=3,  ratio=80, ylim =[-0.1,1.1], legend =:outerbottom, yticks=false)
# 	scatter!((1:truet0)[D[1:truet0] .== 0], D[1:truet0][D[1:truet0] .== 0], ms=3,  label="", m=:x, c=1)
# 	scatter!(((truet0+1):T)[D[(truet0+1):T] .== 1], D[(truet0+1):end][D[(truet0+1):T] .== 1], label=L"C_2"*" Tosses", c=2, ms=3)
# 	scatter!(((truet0+1):T)[D[(truet0+1):T] .== 0], D[(truet0+1):end][D[(truet0+1):T] .== 0], m = :x, c=2, label="", ms=3)
# 	# plot!((truet0+1):T,D[(truet0+1):end], label="C2 Tosses")
# 	vline!([truet0], lw=3, lc=3, label="switching point")
end;

# ╔═╡ 0bfe48f9-f5a6-4bf0-be34-2dfae7ffd988
function bernoulli_pdf(p, y)
	Bool(y) ? p : 1-p
end

# ╔═╡ 8297ae71-ac77-4549-8d9a-2dcbecf6f66b
function ℓ_switch(D, p₁=0.5, p₂=0.2)
	likes = zeros(length(D)-1)
	for t in 1:length(likes)
		likes[t] = prod(bernoulli_pdf.(p₁, D[1:t])) * prod(bernoulli_pdf.(p₂, D[(t+1):end]))
	end
	return likes, sum(likes)
end

# ╔═╡ a373abb9-0003-424b-8675-4c704dbe98c8
function posteriorT0(D, p₁, p₂)
	N = length(D)
	ℓs, sumℓ = ℓ_switch(D, p₁, p₂)
	pt0 = ℓs / sumℓ
	# sample 1:N-1 based on pt0; a method from StatsBase
	# you are allowed to use sampling method off the shelf
	return pt0, sample(1:N-1, Weights(pt0))
end

# ╔═╡ 3a79e32a-39e2-4fa4-bb7b-bd0fe4e83326
function ℓ(θ, 𝒟; logprob=false)
	N = length(𝒟)
	N⁺ = sum(𝒟)
	ℓ(θ, N, N⁺; logprob=logprob)
end

# ╔═╡ 4eaaadfa-32e0-4672-91e4-2a1d53f0e9a1
function ℓ(θ, n, n⁺; logprob=false)
	# logL = n⁺ * log(θ) + (n-n⁺) * log(1-θ)
	# use xlogy(x, y) to deal with the boundary case 0*log(0) case gracefully, i.e. θ=0, n⁺ = 0
	logL = xlogy(n⁺, θ) + xlogy(n-n⁺, 1-θ)
	logprob ? logL : exp(logL)
end

# ╔═╡ 34a24da9-1c4e-42b6-83f5-378a844ae77d
function posteriorP_robust(𝒟, θs)
	likes = [ℓ(θ, 𝒟; logprob=true) for θ in θs]
	logsum = logsumexp(likes)
	posterior_dis = exp.(likes .- logsum)
	return posterior_dis, sample(θs, Weights(posterior_dis))
end

# ╔═╡ 22c558e9-d460-476f-8126-479034ed1ff9
begin

	function gibbsChangePoint(D, mc, burnin; a0=1, b0=1, a1=1, b1=1, discreteP=false, step=0.1)
		T = length(D)
		# initial guess of t0
		t0 = Int(floor(T/2))
		# initial guess of p1
		p1 = mean(D[1:t0])
		# initial guess of p2
		p2 = mean(D[(t0+1):end])	

		ps = 0:step:1.0
		samples = zeros(3, mc) 
		for i in 1:(mc+burnin)
			# sample T0
			_, t0 = posteriorT0(D, p1, p2)
			if discreteP 
				# sample P1
				_,p1 = posteriorP_robust(D[1:t0], ps)
				# sample P2
				_,p2 = posteriorP_robust(D[t0+1:end], ps)
			else
				# if you use continuous random variable
				# we sample Beta distribution directly
				# sample P1
				nh₁ =  sum(D[1:t0])
				p1 = rand(Beta(a0+ nh₁, b0 + (t0-nh₁)))
				# sample P2
				nh₂ =  sum(D[(t0+1):end])
				p2 = rand(Beta(a1+ nh₂, b1+ (T-t0-nh₂)))
			end
			
			if i > burnin
				samples[:, i-burnin] = [t0, p1, p2]
			end
		end
		return samples
	end

end

# ╔═╡ 42cc02f9-179a-436a-bb31-4b8ea7414dd6
begin
	Random.seed!(111)
	samples_dis = gibbsChangePoint(D, 2000, 1000; discreteP=true, step = 0.005)
end;

# ╔═╡ de054b00-4ceb-4e20-a7d2-1efc774d648c
let
	
	plt = plot(1:truet0,D[1:truet0], label="C1 Tosses")
	plot!((truet0+1):T,D[(truet0+1):end], label="C2 Tosses", yticks =[0,1],  xlabel="tosses")
	vline!([truet0], lw=4, label="switching point")
	t0s = samples_dis[1, 1:10:end]
	# scatter!(t0s, 0.5 *ones(length(t0s)), m=:vline, markersize=10, label="Gibbs t0", legend=:outerbottom)

	plt
end

# ╔═╡ d08d890f-704d-4577-b930-7c6e3a31b8dc
sampleChains_discrete = Chains(samples_dis', [:"S", :"P1", "P2"])

# ╔═╡ cab0dbbc-6f1c-4a18-bd7b-f7fe70eaa949
describe(sampleChains_discrete)

# ╔═╡ ac34cd2b-6f06-4998-b7da-04ab889be693
plot(sampleChains_discrete)

# ╔═╡ 79e02757-cd03-46a8-862f-5549da5d03eb
begin
	
	plot(1:truet0,D[1:truet0], label="C1 Tosses")
	plot!((truet0+1):T,D[(truet0+1):end], label="C2 Tosses")
	vline!([truet0], lw=4, label="switching point")
	t0s = samples_dis[1, 1:10:end]
	scatter!(t0s, 0.5 *ones(length(t0s)), m=:vline, markersize=10, label="Gibbs t0", legend=:outerbottom)
end

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
HypertextLiteral = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
LaTeXStrings = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
Latexify = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
LogExpFunctions = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
MCMCChains = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
StatsBase = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
DataFrames = "~1.5.0"
Distributions = "~0.25.113"
HypertextLiteral = "~0.9.4"
LaTeXStrings = "~1.3.0"
Latexify = "~0.16.1"
LogExpFunctions = "~0.3.28"
MCMCChains = "~6.0.6"
Plots = "~1.40.7"
PlutoTeachingTools = "~0.2.13"
PlutoUI = "~0.7.52"
StatsBase = "~0.34.3"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "8ac5d8bb7597907a97f40aac737d568e02e3bdb1"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractMCMC]]
deps = ["BangBang", "ConsoleProgressMonitor", "Distributed", "FillArrays", "LogDensityProblems", "Logging", "LoggingExtras", "ProgressLogging", "Random", "StatsBase", "TerminalLoggers", "Transducers"]
git-tree-sha1 = "aa469a7830413bd4c855963e3f648bd9d145c2c3"
uuid = "80f14c24-f653-4e6a-9b94-39d6b0f70001"
version = "5.6.0"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AbstractTrees]]
git-tree-sha1 = "2d9c9a55f9c93e8887ad391fbae72f8ef55e1177"
uuid = "1520ce14-60c1-5f80-bbc7-55ef81b5835c"
version = "0.4.5"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "b392ede862e506d451fc1616e79aa6f4c673dab8"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.38"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

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

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

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

[[deps.AxisArrays]]
deps = ["Dates", "IntervalSets", "IterTools", "RangeArrays"]
git-tree-sha1 = "16351be62963a67ac4083f748fdb3cca58bfd52f"
uuid = "39de3d68-74b9-583c-8d2d-e117c070f3a9"
version = "0.4.7"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

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
version = "1.3.0+1"

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "ea32b83ca4fefa1768dc84e504cc0a94fb1ab8d1"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.2"

[[deps.ConsoleProgressMonitor]]
deps = ["Logging", "ProgressMeter"]
git-tree-sha1 = "3ab7b2136722890b9af903859afcf457fa3059e8"
uuid = "88cd18e8-d9cc-4ea6-8889-5259c0d15c8b"
version = "0.1.2"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"
weakdeps = ["IntervalSets", "LinearAlgebra", "StaticArrays"]

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrettyTables", "Printf", "REPL", "Random", "Reexport", "SentinelArrays", "SnoopPrecompile", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "aa51303df86f8626a962fccb878430cdb0a97eee"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.5.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

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

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

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
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

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

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "532f9126ad901533af1d4f5c198867227a7bb077"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+1"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "ee28ddcd5517d54e417182fec3886e7412d3926f"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.8"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "f31929b9e67066bee48eec8b03c0df47d31a74b3"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.8+0"

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

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ae350b8225575cc3ea385d4131c81594f86dfe4f"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.12"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "401e4f3f30f43af2c8478fc008da50096ea5240f"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.3.1+0"

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

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

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
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.IntervalSets]]
git-tree-sha1 = "dba9ddf07f77f60450fe5d2e2beb9854d9a49bd0"
uuid = "8197267c-284f-5f27-9208-e0e47529a953"
version = "0.7.10"
weakdeps = ["Random", "RecipesBase", "Statistics"]

    [deps.IntervalSets.extensions]
    IntervalSetsRandomExt = "Random"
    IntervalSetsRecipesBaseExt = "RecipesBase"
    IntervalSetsStatisticsExt = "Statistics"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "0dc7b50b8d436461be01300fd8cd45aa0274b038"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.0"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterTools]]
git-tree-sha1 = "42d5f897009e7ff2cf88db414a389e5ed1bdd023"
uuid = "c8e1da08-722c-5040-9ed9-7db0dc04731e"
version = "1.10.0"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "36bdbc52f13a7d1dcb0f3cd694e01677a515655b"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+0"

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
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LeftChildRightSiblingTrees]]
deps = ["AbstractTrees"]
git-tree-sha1 = "fb6803dafae4a5d62ea5cab204b1e657d9737e7f"
uuid = "1d6d02ad-be62-4b6b-8a6d-2f90e265016e"
version = "0.2.0"

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
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "edbf5309f9ddf1cab25afc344b1e8150b7c832f9"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.LogDensityProblems]]
deps = ["ArgCheck", "DocStringExtensions", "Random"]
git-tree-sha1 = "4e0128c1590d23a50dcdb106c7e2dbca99df85c0"
uuid = "6fdf6af0-433a-55f7-b3ed-c6c6e0b8df7c"
version = "2.1.2"

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

[[deps.MCMCChains]]
deps = ["AbstractMCMC", "AxisArrays", "Dates", "Distributions", "IteratorInterfaceExtensions", "KernelDensity", "LinearAlgebra", "MCMCDiagnosticTools", "MLJModelInterface", "NaturalSort", "OrderedCollections", "PrettyTables", "Random", "RecipesBase", "Statistics", "StatsBase", "StatsFuns", "TableTraits", "Tables"]
git-tree-sha1 = "d28056379864318172ff4b7958710cfddd709339"
uuid = "c7f686f2-ff18-58e9-bc7b-31028e88f75d"
version = "6.0.6"

[[deps.MCMCDiagnosticTools]]
deps = ["AbstractFFTs", "DataAPI", "DataStructures", "Distributions", "LinearAlgebra", "MLJModelInterface", "Random", "SpecialFunctions", "Statistics", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "4e1d6c34e22ee75892f9b371494ec98e8a6bf46a"
uuid = "be115224-59cd-429b-ad48-344e309966f0"
version = "0.3.12"

[[deps.MIMEs]]
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0eef589dd1c26a3ac9d753fe1a8bcad63f956fa6"
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.16.8+1"

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NaturalSort]]
git-tree-sha1 = "eda490d06b9f7c00752ee81cfa451efe55521e21"
uuid = "c020b1a1-e9b0-503a-9c33-f039bfc54a85"
version = "1.0.0"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

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
version = "0.3.29+0"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.7+0"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

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
version = "10.44.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "e127b609fb9ecba6f201ba7ab753d5a605d53801"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.54.1+0"

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
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "f202a1ca4f6e165238d8175df63a7e26a51e04dc"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.7"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressLogging]]
deps = ["Logging", "SHA", "UUIDs"]
git-tree-sha1 = "80d919dee55b9c50e8d9e2da5eeafff3fe58b539"
uuid = "33c8b6b6-d38a-422a-b730-caa89a2f386c"
version = "0.1.4"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

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

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

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

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

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
version = "1.12.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "2f5d4697f21388cbe1ff299430dd169ef97d7e14"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.4.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

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

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

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
git-tree-sha1 = "5cf7606d6cef84b543b483848d4ae08ad9832b21"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.3"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

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

[[deps.TerminalLoggers]]
deps = ["LeftChildRightSiblingTrees", "Logging", "Markdown", "Printf", "ProgressLogging", "UUIDs"]
git-tree-sha1 = "f133fab380933d042f6796eda4e130272ba520ca"
uuid = "5d786b92-1e48-4d6f-9151-6b4477ca9bed"
version = "0.1.7"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

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

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "d95fe458f26209c66a187b1114df96fd70839efd"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.0"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

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
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "15e637a697345f6743674f1322beefbc5dcd5cfc"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+0"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

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

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

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
version = "1.3.1+2"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "555d1076590a6cc2fdee2ef1469451f872d8b41b"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+1"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "3516a5630f741c9eecb3720b1ec9d8edc3ecc033"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+0"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

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
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

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

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"

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
# ╟─71d93c0d-0e4a-4684-8a54-6e25c2606a2b
# ╟─428492c0-8ee4-4a48-938b-fa2f4fc93964
# ╟─ff753e14-2025-4a94-8faf-d5368f27a8f9
# ╟─797629b6-0669-11ec-3778-d1f20c3d6a4c
# ╟─6374b212-ecf4-4887-9e94-fdbb24ca5661
# ╟─c7068d15-80be-4a1e-8cbf-f8c62f777b0f
# ╟─72a517ce-6d49-4c96-b334-52184f98f296
# ╟─5be1a24e-f3a5-4672-9b0b-baa116886567
# ╟─23a7ed25-9c38-4eb2-b4ad-32e0e38c70a8
# ╟─f1d1b12e-3d73-469c-b23d-1534475c07a5
# ╟─6414fac6-de38-4a71-aacc-76c6df248225
# ╟─cfab05b9-09ff-463c-9e52-c86a35e7a6ea
# ╟─afe11239-08b6-4053-88e4-8795dc62ac04
# ╟─bf204cbd-386f-4456-b1d4-38106826a56d
# ╟─32e042d4-cbe7-40e4-b479-1896ba38a436
# ╟─58c2e8d0-dba2-4884-9250-f2aa50af37e1
# ╟─6658512a-f80f-4ef7-b927-2fee89e670e3
# ╟─b999fe4b-8c22-4844-a9b7-15d7cc0052f2
# ╟─283871df-944a-4223-a7e1-043080542843
# ╟─8cfe4da4-4cd9-4d75-9446-b5bc1cc122fd
# ╟─0db3a069-725a-4d40-a1b7-e16aa2fbc5f2
# ╟─6009b022-8bc6-443b-87b0-6f75db590da2
# ╟─93a14c73-5039-4099-9068-60c25b20accd
# ╟─a584b032-e693-48c6-8242-f408187168c6
# ╟─61f6e84a-25dc-4def-9dc3-5faa1392dfa7
# ╟─1a944206-1c2e-47c7-8083-e28e9f60bd7d
# ╟─d5039ee8-2e0a-4b66-aae3-355d3d7f40d3
# ╟─b12ae2b7-01e1-4dee-b217-259651a360e2
# ╟─e9778657-476e-4920-9ffe-596dc283da80
# ╟─8f5c045c-3191-4031-92fd-c8975b0e3a35
# ╟─bacfddfa-bf66-4ad1-8c7d-f53df38ccd67
# ╟─3e24cb61-fbc6-4e68-8d51-10447544b1ad
# ╟─ca522196-9709-4e55-b7d7-62c90f679453
# ╟─d432f301-c487-4b47-aebb-72b9bff7c7b1
# ╟─361d1d0e-949d-40a5-afbd-a32a052d426e
# ╟─966e5687-af01-4b2d-afd3-5a2929420d94
# ╟─49f7774e-14e0-49fb-985c-fa7f5f07f6c8
# ╟─1b0f0a64-8868-41af-b609-028ae3090780
# ╟─9f0ed17d-0f14-47cf-ac89-b6803a7174a1
# ╠═9cb4bec9-b655-433c-8ebc-237a5fbbd0e9
# ╠═542856d2-41df-49e2-9ee1-d02968a93f5e
# ╠═f7e4846c-be4c-43ad-905c-6662afd30981
# ╟─5671cf5c-e5a8-45eb-8bb2-b021ff98409e
# ╟─b64720e2-8be7-42e9-a7ce-4a049520ac25
# ╟─258d0224-650e-44e5-a28c-43ab141c5053
# ╟─6582f9c8-9161-4dec-9401-24487b26c0f3
# ╟─d469deda-38d5-43bb-b638-3b058c046060
# ╟─fd78a28e-ff0c-4c94-be1e-8870b1b36060
# ╟─502598df-9e7a-4887-aa76-ee52368f9bcd
# ╟─564dfbdb-ebdd-432c-b3c9-29d429360c28
# ╟─a4b5e04b-2be1-4ad8-8025-94957e98623a
# ╟─18f00394-153f-4260-b5ec-3c26aad32eb9
# ╟─d086fce9-74ea-4f22-9ec9-84c12aca7867
# ╟─ef2bcd80-8975-42d7-a7da-15045d575047
# ╟─a66da406-fc81-47bc-8745-db1959340ad9
# ╟─3e2a88ec-138b-4ec5-a87d-bb2a876fd308
# ╟─818ade09-6830-45d5-ae0c-ed73c76c656c
# ╟─b54db267-1c15-46f5-9b9d-a951a5d9ae0f
# ╟─de054b00-4ceb-4e20-a7d2-1efc774d648c
# ╟─c11b8f03-b0ba-41c5-8b6d-8b2e6fb4489c
# ╟─0bfe48f9-f5a6-4bf0-be34-2dfae7ffd988
# ╟─8297ae71-ac77-4549-8d9a-2dcbecf6f66b
# ╟─a373abb9-0003-424b-8675-4c704dbe98c8
# ╟─72aad7ff-6ed2-4469-8937-93a75b038c0f
# ╟─9fc16fb3-49c2-47ee-b9a2-1a9b59b0e9e5
# ╟─34a24da9-1c4e-42b6-83f5-378a844ae77d
# ╟─3a79e32a-39e2-4fa4-bb7b-bd0fe4e83326
# ╟─4eaaadfa-32e0-4672-91e4-2a1d53f0e9a1
# ╟─22c558e9-d460-476f-8126-479034ed1ff9
# ╟─42cc02f9-179a-436a-bb31-4b8ea7414dd6
# ╟─30e4e84b-39dc-4209-b77e-1ae561b173f9
# ╠═d08d890f-704d-4577-b930-7c6e3a31b8dc
# ╠═cab0dbbc-6f1c-4a18-bd7b-f7fe70eaa949
# ╟─0db15737-2193-46a1-8266-070b4d2b7693
# ╠═ac34cd2b-6f06-4998-b7da-04ab889be693
# ╟─79e02757-cd03-46a8-862f-5549da5d03eb
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
