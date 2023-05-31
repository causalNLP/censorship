from src import ShapleyScorerFactory, TrainerFactory

def finetune():
    trainer = TrainerFactory.create("default",
                                    dataset_path="data/dataset.csv",
    )
    trainer.train(output_dir="models",
                  learning_rate=1e-5,
                  epochs=10,
                  batch_size=32,
                  weight_decay=0.01)
    
if __name__ == "__main__":
    finetune()