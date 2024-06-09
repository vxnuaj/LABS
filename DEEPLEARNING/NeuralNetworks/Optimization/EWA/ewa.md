$V_t = \beta V_{t-1} + (1 - \beta) \theta_t$


$V_{10} = (.05)\theta_{10} + (.95)V_{9}$

$V_{10} = (.05)\theta_{10} + (.95)(.05(\theta_9) + (.95^2)V_8)$

$V_{10} = (.05)\theta_{10} + (.95)(.05(\theta_9) + (.95^2)(.05(\theta_8) + (.95^3)V_7))$

**Bias Correction:**

$v_{t} = \frac{v_t}{1-\beta}$