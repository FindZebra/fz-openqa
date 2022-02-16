CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python run.py +experiment=option_retriever +environ=titan logger.wandb.name=colbert-iw-reinforce-qexp-200 trainer.max_epochs=30


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python run.py +experiment=option_retriever +environ=titan logger.wandb.name=dpr-iw-reinforce-qexp-200 +patch=dpr trainer.max_epochs=30


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 poetry run python run.py +experiment=option_retriever +environ=titan logger.wandb.name=colbert-in-batch-qexp-200 trainer.max_epochs=40 model.module.grad_expr=in_batch
