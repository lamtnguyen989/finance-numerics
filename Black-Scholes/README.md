# Black-Scholes model

## Geometric Brownian Motion and Black-Scholes derivation

The stochastic dynamics that underlying the Black-Scholes option pricing model is that of the Geometric Brownian Motion (GBM) or the exponential return price process $dS_t = \mu S_t dt + \sigma S_t dB_t$. The reasoning behind the latter title is under the Itô formula (using alongside the heuristic rule of thumb $(dB_t)^2 \to dt$), the logarithmic differential can be computed as 

$$\begin{align*}
	d(\ln S_t) & = \left( \frac{\partial \ln S_t}{\partial t} + \frac{(\sigma S_t)^2}{2}\frac{\partial^2 \ln S_t}{\partial S_t^2} \right)dt + \frac{\partial \ln S_t}{\partial S_t}dS_t \\
	& = -\frac{\sigma^2S_t^2}{2}\frac{1}{S_t^2} dt + \frac{1}{S_t}(\mu S_t dt + \sigma S_t dB_t) \\
	& = -\frac{\sigma^2}{2} dt +\mu dt + \sigma dB_t \\
	& = \left(\mu -\frac{\sigma^2}{2}\right)dt + \sigma dB_t
\end{align*}$$

so assuming $t_0 = 0$ and $B_0 = 0$, we can integrate to get 

$$\begin{align*}
	\int_0^t d(\ln S_\tau) = \int_0^t \left(\mu -\frac{\sigma^2}{2}\right)d\tau + \int_0^t\sigma dB_\tau \implies \ln \frac{S_t}{S_0} = \left(\mu -\frac{\sigma^2}{2}\right)t + \sigma B_t
\end{align*}$$

therefore exponentiating gives 

$$\begin{align*}
	S_t = S_0\exp\left[\left(\mu -\frac{\sigma^2}{2}\right)t + \sigma B_t \right]
\end{align*}$$.

In particular, the Black-Scholes model assumes that the stock price $S$ follows the GBM and so its time-based value then can be modeled as the stochastic process $V(S,t)$ where under the Itô formula

$$\begin{align*}
	dV = \left(\frac{\partial V}{\partial t} + \mu S\frac{\partial V}{\partial S} + \frac{(\sigma S)^2}{2}\right)dt + \sigma S\frac{\partial V}{\partial S}dB
\end{align*}$$

So we can eliminate the stochastic term by considering the self-financing portfolio $\Pi = \frac{\partial V}{\partial S}S - V$ since

$$\begin{align*}
	d\Pi & = \frac{\partial V}{\partial S}dS - dV \\
	& = \frac{\partial V}{\partial S}(\mu S dt + \sigma S dB) - \left(\frac{\partial V}{\partial t} + \mu S\frac{\partial V}{\partial S} + \frac{(\sigma S)^2}{2}\right)dt - \sigma S\frac{\partial V}{\partial S}dB \\
	& =  -\left(\frac{\partial V}{\partial t}+ \frac{(\sigma S)^2}{2}\right)dt
\end{align*}$$

so since the portfolio is riskless, we can assume a risk-free return rate $r$ and so we can model the change of this portfolio as $d\Pi = r\Pi dt$, after subtituting in values, we see that 

$$\begin{align*}
	-\left(\frac{\partial V}{\partial t}+ \frac{(\sigma S)^2}{2}\right)dt = r\left(S\frac{\partial V}{\partial S} - V\right)dt \implies -\left(\frac{\partial V}{\partial t}+ \frac{(\sigma S)^2}{2} -r\left(S\frac{\partial V}{\partial S} - V\right)\right)dt = 0
\end{align*}$$

so since $dt>0$, we arrive at the infamous PDE,

$$\begin{align*}
	\frac{\partial V}{\partial t}+ \frac{\sigma^2 S^2}{2} -rS\frac{\partial V}{\partial S} - rV= 0
\end{align*}$$