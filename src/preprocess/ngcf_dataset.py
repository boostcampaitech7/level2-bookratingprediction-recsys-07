from torch.utils.data import Dataset


class GraphDataset(Dataset):
    def __init__(self, dataframe):
        super(Dataset, self).__init__()

        self.uid = list(dataframe["user_id"])
        self.iid = list(dataframe["item_id"])
        self.ratings = list(dataframe["rating"])

    def __len__(self):
        return len(self.uid)

    def __getitem__(self, idx):
        uid = self.uid[idx]
        iid = self.iid[idx]
        rating = self.ratings[idx]

        return uid, iid, rating
