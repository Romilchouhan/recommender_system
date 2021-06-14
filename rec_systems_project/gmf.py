import torch

class GMF(torch.nn.Module):
    def __init__(self, num_users, num_items, latent_dim):
        super(GMF, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = latent_dim

        self.embedding_user = torch.nn.Embedding(num_embedding=self.num_users, embedding_dim=self.latent_dim, max_norm=True)
        self.embedding_item = torch.nn.Embedding(num_embedding=self.num_items, embedding_dim=self.latent_dim, max_norm=True)
        
        
        self.linear = torch.nn.Linear(in_features=self.latent_dim, out_features=1)
        self.logistic = torch.nn.Sigmoid()

    def forward(self, user_indices, item_indices):
        user_embedding = self.embedding_user(user_indices)
        item_embedding = self.embedding_item(item_indices)
        element_wise_product = torch.mul(user_embedding, item_embedding)
        logits = self.linear(element_wise_product)
        rating = self.logistic(logits)
        return rating

