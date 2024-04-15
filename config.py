# Description: Configuration file for PSID

SEED_VALUE = 42
TEST_SIZE = 0.5
N_SAMPLE = 1
NUM_BINS = 15

#Best trial config: {'n_epochs': 100, 'lr': 0.00016965472509281913, 'batch_size': 32, 'weight_decay': 0.014459457087382013, 'embedding_dim': 512, 'n_head': 32, 'dropout_rate': 0.010097730750257106}
#Best trial final t_hat validation loss: 0.03109976978573416
#PSID config

N_EPOCHS = 100
LOG_FEQ = 10
BATCH_SIZE = 32
LEARNING_RATE = 0.00016965472509281913
WEIGHT_DECAY = 0.01
DROPOUT_RATE = 0.010097730750257106
EMBEDDING_DIM = 512
N_HEAD = 32


#Best trial config: {'n_epochs': 100, 'lr': 0.0005438095887661997, 'batch_size': 128, 'weight_decay': 0.008629975532304214, 'embedding_dim': 512, 'n_head': 16, 'dropout_rate': 0.007872866455733318}
#Best trial final t_hat validation loss: 0.03981799360917648
#CPS config
# tunning hyperparameters
'''
N_EPOCHS = 100
LOG_FEQ = 10
BATCH_SIZE = 128
LEARNING_RATE = 0.0005438095887661997
WEIGHT_DECAY = 0
DROPOUT_RATE = 0
EMBEDDING_DIM = 512
N_HEAD = 16
'''

#Best trial config: {'n_epochs': 20, 'lr': 0.001, 'batch_size': 128, 'weight_decay': 0.1, 'embedding_dim': 512, 'n_head': 16, 'dropout_rate': 0.1}
#Best trial final t_hat validation loss: 0.45675197719259464
#Twins config
'''
N_EPOCHS = 5
LOG_FEQ = 1
BATCH_SIZE = 32
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0
DROPOUT_RATE = 0.05
EMBEDDING_DIM = 512
N_HEAD = 16
'''
