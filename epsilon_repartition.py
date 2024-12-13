import numpy as np
import matplotlib.pyplot  as plt
def epsilon_rep(episode):
    return  max(0.01, (0.9997 ** episode))

epsilon_rep = np.vectorize(epsilon_rep)
les_x = np.linspace(0, 10_000, 10_000)
les_y = epsilon_rep(les_x)
plt.plot(les_x, les_y, '.')
plt.show()

