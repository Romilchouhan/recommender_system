class MovielensTrainDataset(Dataset):
    def __init__(self, data, movie_ids):
        self.data = data
        self.movie_ids = movie_ids
        self.users, self.movies, self.labels = self.get_dataset(data, movie_ids)
    
    def __getitem__(self, idx):
        return self.users[idx], self.movies[idx], self.labels[idx]
    
    def __len__(self):
        return len(self.users)
    
    def get_dataset(self, data, movie_ids):
        users, movies, labels = [], [], []

        # This is the set of items that each user has interaction with
        user_item_set = set(zip(train_data['user_id'], train_data['movie_id']))

        # 4:1 ratio of negative to positive samples
        num_negatives = 4

        for (user, movie) in tqdm(user_item_set):
            users.append(user)
            movies.append(movie)
            labels.append(1)
            for _ in range(num_negatives):
                # randomly select a movie
                negative_movie = np.random.choice(movie_ids)
                # check that the user has not interacted with this item
                while (user, negative_movie) in user_item_set:
                    negative_movie = np.random.choice(movie_ids)
                users.append(user)
                movies.append(negative_movie)
                labels.append(0)  # items not interacted are negative
            return torch.tensor(users), torch.tensor(movies), torch.tensor(labels)
