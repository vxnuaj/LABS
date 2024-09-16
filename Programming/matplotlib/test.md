P1: 

x = [1, 2, 3, 4, 5]
y = [2, 3, 5, 7, 11]

plt.plot(x, y)
ax = plt.gca()
ax.set_title('Prime Numbers Plot', fontsize = 'xx-large')
ax.set_xlabel('X-axis', fontsize = 'large')
ax.set_ylabel('Y-axis', fontsize = 'large')


P2:

x = [1, 2, 3, 4, 5]
y = [2, 3, 6, 8, 10]

plt.scatter(x, y, label = 'Data Points', c = 'red')
plt.legend()

P3

categories = ['A', 'B', 'C', 'D']
values = [10, 20, 15, 25]

plt.barh(y = values, left = categories, width = 5)
plt.gca().set_xlabel('Category Values')
plt.gca().set_ylabel('Labels')

P4

plt.close('all')

data = np.random.normal(size = (1000,), loc = 0, scale = 1)

plt.hist(data, bins = 30)
plt.suptitle('Histogram of Normally Distributed Data', fontsize = 'xx-large')

P5

plt.close('all')

x = [0, 1, 2, 3, 4, 5]
y = [0, 1, 4, 9, 16, 25]

plt.plot(x, y)
ax = plt.gca()
ax.set_xlabel('X-axis', fontsize = 'x-large')
ax.set_ylabel('Y-axis', fontsize = 'x-large')

Y and X axis already show ticks at intervals of 5 and 1 respectivel. Will change them to show  minor ticks instead

plt.minorticks_on()

P6

plt.close('all')

x = [1, 2, 3, 4]
y = [1, 2, 3, 4]

ax1 = plt.subplot(2, 2, 1)
ax2 = plt.subplot(2, 2, 2)
ax3 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

ax1.plot(x, y)
ax2.scatter(x, y)
ax3.bar(x, 5)
ax4.hist(x, bins = 4)

plt.gcf().suptitle('Data')
plt.gcf().tight_layout()

plt.show()


P7

plt.close('all')

x = [1, 2, 3, 4]
y = [1, 2, 3, 4]
x_hist = [1,1,1, 2, 3, 3, 4, 4, 4, 4]

fig, axs = plt.subplot_mosaic('''
                   AB
                   CD''', layout = 'constrained')


axs['A'].plot(x, y)
axs['A'].set_title('Line Plot')
axs['B'].scatter(x, y)
axs['B'].set_title('Scatter Plot')
axs['C'].bar(x, height = 5)
axs['C'].set_title('Bar Chart')
axs['D'].hist(x_hist, bins = 4)
axs['D'].set_title('Histogram')


plt.show()

P8

x = [1, 2, 3, 4]
y = [4, 3, 2, 1]$
 
plt.plot(x, y)
plt.savefig('figures/line_plot.png', dpi = 300)

P9

np.random.seed(2)

data = {'A': np.random.randn(10), 'B': np.random.randn(10)}

df = pd.DataFrame(data)
df.plot()

ax = plt.gca()
ax.set_title('np.random.randn plot')
ax.set_ylabel('Y-axis')
ax.set_xlabel('X-axis')

P10

plt.close('all')

data = {'X': np.arange(start = 1, stop = 11), 'Y': np.square(np.arange(start = 1, stop = 11))}
df = pd.DataFrame(data)
line = plt.plot(data['X'], data['Y'], color = 'green')

