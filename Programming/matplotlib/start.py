import matplotlib.pyplot as plt
import numpy as np

fig = plt.figure(figsize =(7, 7), layout = 'constrained')
fig.suptitle('D2L', fontsize = 'xx-large', fontweight = 'bold')
fig.supxlabel(t = 'X',  fontsize = 'xx-large', fontweight = 'bold')
fig.supylabel(t = 'Y', fontsize = 'xx-large', fontweight = 'bold')

x, y = np.array([1, 2, 3, 4]), np.array([3, 4, 5, 6])

ax0 = fig.add_subplot(2, 2, 1)
ax0.plot(x, y, label = 'hello')
ax0.plot(x, y[::-1], label = 'goodbye')
ax0.set_title('1', fontsize = 'large', fontweight = 'bold')
ax0.set_ylabel(r'$\sigma1$', fontsize = 'large', fontweight = 'bold')
ax0.set_xlabel(r'$\theta1$',  fontsize = 'large', fontweight = 'bold')
ax0.grid(True)
ax0.legend()

ax1 = fig.add_subplot(2, 2, 2)
ax1.set_title('2', fontsize = 'large', fontweight = 'bold')
ax1.set_ylabel(r'$\sigma2$', fontsize = 'large', fontweight = 'bold')
ax1.set_xlabel(r'$\theta2$', fontsize = 'large', fontweight = 'bold')
ax1.plot(x, y)

ax2 = fig.add_subplot(2, 2, 3)
ax2.set_title('3', fontsize = 'large', fontweight = 'bold')
ax2.set_ylabel(r'$\sigma3$', fontsize = 'large', fontweight = 'bold')
ax2.set_xlabel(r'$\theta3$', fontsize = 'large', fontweight = 'bold')
ax2.plot(x, y)

ax3 = fig.add_subplot(2, 2, 4)
ax3.set_title('4', fontsize = 'large', fontweight = 'bold')
ax3.set_ylabel(r'$\sigma4$', fontsize = 'large', fontweight = 'bold')
ax3.set_xlabel(r'$\theta4$', fontsize = 'large', fontweight = 'bold')
ax3.plot(x, y)

plt.show()