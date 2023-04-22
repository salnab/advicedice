import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing

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
            "users": torch.tensor(user, dtype=torch.long),
            "games": torch.tensor(game, dtype=torch.long),
            "ratings": torch.tensor(rating, dtype=torch.float),
        }
    
# write a function that takes in a dataframe and returns a GameDataset object
def create_dataset(df):
    users = df["user_id"].values
    games = df["game_id"].values
    ratings = df["rating"].values
    return GameDataset(users, games, ratings)

def monitor_error(self, output, rating):
    output = output.detach().cpu().numpy()
    rating = rating.detach().cpu().numpy()
    return {"rmse": np.sqrt(mean_squared_error(output, rating))}

#write a class called RecSysModel that build a neural network with an embedding layer
class RecSysModel(nn.Module):
    def __init__(self, n_users, n_games):
        super(RecSysModel, self).__init__()
        self.user_embed = nn.Embedding(n_users, 32)
        self.game_embed = nn.Embedding(n_games, 32)
        self.user_bias = nn.Embedding(n_users, 1)
        self.game_bias = nn.Embedding(n_games, 1)
        self.fc = nn.Linear(64, 1)
        self.step_scheduler_after = "epoch"
        self.train_one_epoch = self.train_one_epoch
        self.eval_one_epoch = self.eval_one_epoch

    def fit(self, train_dataset, val_dataset, epochs=5, batch_size=1024):
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        optimizer = self.fetch_optimizer()
        scheduler = self.fetch_scheduler(optimizer)
        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            print("-" * 10)
            train_loss = self.train_one_epoch(train_loader, optimizer)
            print(f"Train loss {train_loss}")
            val_loss, eval_metrics = self.eval_one_epoch(val_loader)
            print(f"Val loss {val_loss}")
            print(f"Eval metrics {eval_metrics}")
            scheduler.step()
            print()

    def fetch_optimizer(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
    
    def fetch_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
    
    def forward(self, users, games, ratings=None):
        user_vecs = self.user_embed(users)
        game_vecs = self.game_embed(games)
        user_bias = self.user_bias(users)
        game_bias = self.game_bias(games)
        output = torch.cat([user_vecs, game_vecs], dim=1)
        output = self.fc(output)
        output = output + user_bias + game_bias
        if ratings:
            loss = nn.MSELoss()(output, ratings.view(-1, 1))
            eval_metrics = self.monitor_metrics(output, ratings.view(-1, 1))
            return output, loss, eval_metrics
        
def train(df):
    lbl_user = preprocessing.LabelEncoder()
    lbl_game = preprocessing.LabelEncoder()
    df["user_id"] = lbl_user.fit_transform(df["user_id"].values)
    df["game_id"] = lbl_game.fit_transform(df["game_id"].values)

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)
    train_dataset = create_dataset(train_df)
    val_dataset = create_dataset(val_df)

    model = RecSysModel(n_users=len(lbl_user.classes_), n_games=len(lbl_game.classes_))
    model.fit(train_dataset, val_dataset, epochs=5)

if __name__ == "__main__":
    df = pd.read_csv("data/sample.csv")
    train(df)




    