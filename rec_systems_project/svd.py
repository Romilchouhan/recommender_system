import numpy as np

class SVDModel(object):
    def __init__(self, num_items, num_users, mean,
                num_factors=100, init_variance=0.1):
        self.mu = mean
        self.num_items = num_items
        self.num_users = num_users
        self.num_factors = num_factors
        # Deviation, per-item
        self.b_i = np.zeros((num_items,))
        # Deviation, per-user:
        self.b_u = np.zeros((num_users,))
        # Factors matrices:
        self.q = np.random.randn(num_factors, num_items) * init_variance
        self.p = np.random.randn(num_factors, num_users) * init_variance
        
    def predict(self, items, users):
        # We don't multiply items and users like matrices here
        # Rather, we just do row-by-row dot products.
        # Matrix multiply would give us every combination of item and user
        # which isn't what we want
        return self.mu + self.b_i[items] + self.b_u[users] + (self.q[:, items] * self.p[:, users]).sum(axis=0)
    
    def error(self, items, users, ratings):
        p = self.predict(items, users)
        d = p - ratings
        rmse = np.sqrt(np.square(d).sum() / items.size)
        mae = np.abs(d).sum() / items.size
        return rmse, mae
    
    def update_by_gradient(self, i, u, r_ui, lambda_, gamma):
        e_ui = r_ui - self.predict(i, u)  # r_ui is the actual value
        dbi = gamma * (e_ui - lambda_ * self.b_u[u])
        dbu = gamma * (e_ui - lambda_ * self.b_i[i])
        dpu = gamma * (e_ui * self.q[:, i] - lambda_ * self.p[:, u])
        dqi = gamma * (e_ui * self.p[:, u] - lambda_ * self.q[:, i])
        self.b_i[i] += dbi
        self.b_u[u] += dbu
        self.p[:,u] += dpu
        self.q[:,i] += dqi
        
    def train(self, items, users, ratings, gamma=0.005, lambda_=0.02,
             num_epochs=20, epoch_callback=None):
        """"Train with Stochastic gradient descent"""
        for epoch in range(num_epochs):
            for idx in np.random.permutation(len(items)):
                i, u, r_ui = items[idx], users[idx], ratings[idx]
                self.update_by_gradient(i, u, r_ui, lambda_, gamma)
            if epoch_callback:
                epoch_callback(self, epoch, num_epochs)
