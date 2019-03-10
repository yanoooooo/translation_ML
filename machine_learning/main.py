from train import TrainManager

epoch = 10

if __name__=="__main__":
    en_data = [("Are you going to Scarborough Fair", "1 1 2 1 3 1")]
    ja_data = [("あの 地方 に 行く の かい", "2 2 1 2 1 1")]

    train_manager = TrainManager()
    train_manager.train(en_data, ja_data, epoch)