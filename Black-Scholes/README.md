# Black-Scholes model

The stochastic dynamics that underlying the Black-Scholes option pricing model is that of the Geometric Brownian Motion (GBM) or the exponential return price process $dS_t = \mu S_t dt + \sigma S_t dB_t$. The reasoning behind the latter title is under the Itô formula (using alongside the heuristic rule of thumb $(dB_t)^2 \to dt$), the logarithmic differential can be computed as 

$$\begin{align*}
	d(\ln S_t) & = \left( \frac{\partial \ln S_t}{\partial t} + \frac{(\sigma S_t)^2}{2}\frac{\partial^2 \ln S_t}{\partial S_t^2} \right)dt + \frac{\partial \ln S_t}{\partial S_t}dS_t \\
	& = -\frac{\sigma^2S_t^2}{2}\frac{1}{S_t^2} dt + \frac{1}{S_t}(\mu S_t dt + \sigma S_t dB_t) \\
	& = -\frac{\sigma^2}{2} dt +\mu dt + \sigma dB_t \\
	& = \left(\mu -\frac{\sigma^2}{2}\right)dt + \sigma dB_t
\end{align*}$$

so assuming $t_0 = 0$ and $B_0 = 0$, we can integrate to get 

$$\begin{align*}
	\int_0^t d(\ln S_\tau) = \int_0^t \left(\mu -\frac{\sigma^2}{2}\right)d\tau + \int_0^t\sigma dB_\tau \implies \ln \frac{S_t}{S_0} & = \left(\mu -\frac{\sigma^2}{2}\right)t + \sigma B_t
\end{align*}$$

therefore exponentiating gives 

$$\begin{align*}
	S_t & = S_0\exp\left[\left(\mu -\frac{\sigma^2}{2}\right)t + \sigma B_t \right]
\end{align*}$$