import pandas as pd
import torch
from sklearn.model_selection import train_test_split

# write a class called GameDataset that instantiates a dataset with user_id, game_id, and rating
class GameDataset:
    def __init__(self, users, games, ratings):
        self.users = users
        self.games = games
        self.ratings = ratings

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, item):
        user = self.users[item]
        game = self.games[item]
        rating = self.ratings[item]
        return {
            "user_id": torch.tensor(user, dtype=torch.long),
            "game_id": torch.tensor(game, dtype=torch.long),
            "rating": torch.tensor(rating, dtype=torch.float),
        }
    
# write a function that takes in a dataframe and returns a GameDataset object
def create_dataset(df):
    users = df["user_id"].values
    games = df["game_id"].values
    ratings = df["rating"].values
    return GameDataset(users, games, ratings)

# write a function that creates a training and validation dataset using train_test_split
def create_train_val(df):
    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)
    return train_dataset, val_dataset

if __name__ == "__main__":
    df = pd.read_csv("data/steam.csv")
    train_dataset, val_dataset = create_train_val(df)




    