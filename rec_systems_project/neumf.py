class NeuMF(pl.LightningModule):
    def __init__(self, num_users, num_items, latent_dim_mf, latent_dim_mlp, data, movie_ids):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim_mf = latent_dim_mf
        self.latent_dim_mlp = latent_dim_mlp
        
        # mf part
        self.embedding_user_mf = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mf)
        self.embedding_item_mf = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mf)
        
        # mlp part
        self.embedding_user_mlp = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim_mlp)
        self.embedding_item_mlp = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim_mlp)
        
        self.fc1 = nn.Linear(in_features=16, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=16)
        self.fc3 = nn.Linear(in_features=16, out_features=8)
        self.logits = nn.Linear(in_features=8+self.latent_dim_mf , out_features=1)
        self.logistic = torch.nn.Sigmoid()
        self.data = data
        self.movie_ids = movie_ids
        
    def forward(self, user_indices, item_indices):
        user_embedding_mf = self.embedding_user_mf(user_indices)
        item_embedding_mf = self.embedding_item_mf(item_indices)
        user_embedding_mlp = self.embedding_user_mlp(user_indices)
        item_embedding_mlp = self.embedding_item_mlp(item_indices)
        
        # mf part
        mf_vector = torch.mul(user_embedding_mf, item_embedding_mf)
        mf_vector = nn.Dropout(p=0.5)(mf_vector)

        # mlp part
        mlp_vector = torch.cat([user_embedding_mlp, item_embedding_mlp], dim=-1)
        m = nn.BatchNorm1d(num_features=16)
        m(mlp_vector)
        mlp_vector = self.fc1(mlp_vector)
        mlp_vector = nn.ReLU()(mlp_vector)
        mlp_vector = nn.Dropout(p=0.5)(mlp_vector)
        
        m = nn.BatchNorm1d(num_features=32)
        m(mlp_vector)
        mlp_vector = self.fc2(mlp_vector)
        mlp_vector = nn.ReLU()(mlp_vector)
        mlp_vector = nn.Dropout(p=0.5)(mlp_vector)
        
        m = nn.BatchNorm1d(num_features=16)
        m(mlp_vector)
        mlp_vector = self.fc3(mlp_vector)
        mlp_vector = nn.ReLU()(mlp_vector)
        mlp_vector = nn.Dropout(p=0.5)(mlp_vector)
        
        
        # concat the mlp and mf part
        vector = torch.cat([mlp_vector, mf_vector], dim=-1)
        logits = self.logits(vector)
        rating = self.logistic(logits)
        return rating
    
    def training_step(self, batch, batch_idx):
        user_indices, item_indices, labels = batch
        predicted_labels = self(user_indices, item_indices)
        criterion = nn.MSELoss()
        loss = torch.sqrt(nn.MSELoss()(predicted_labels, labels.view(-1, 1).float()))
        return loss
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-4)
    
    def train_dataloader(self):
        return DataLoader(MovielensTrainDataset(self.data, self.movie_ids),
                         batch_size=512, num_workers=4)
