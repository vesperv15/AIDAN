import sympy
import os

MATH_TOKENS = [
    'x', 'y', 'z', 't',
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    '+', '-', '*', '/', '**',
    '(', ')',
    'sin', 'cos', 'exp', 'log', 'tan'
]

ACTION_TOKENS = [
    'SIMPLIFY',
    'EXPAND',
    'FACTOR',
    'DIFFERENTIATE_X',
    'TERMINATE',
    'UNDO'
]

TOKEN_LIST = MATH_TOKENS + ACTION_TOKENS + ['<UNK>', '<PAD>']

TOKEN_TO_ID = {token: i for i, token in enumerate(TOKEN_LIST)}
ID_TO_TOKEN = {i: token for i, token in enumerate(TOKEN_LIST)}

VOCAB_SIZE = len(TOKEN_LIST)
ACTION_SIZE = len(ACTION_TOKENS)

EMBED_DIM = 32
HIDDEN_DIM = 128
NUM_LAYERS = 1
FOCUS_DIM=32

GAMMA = 0.99
LR = 0.0003
ENTROPY_BETA = 0.08#0.02
VALUE_LOSS_BETA = 0.5
MAX_STEPS = 30 #burası 50
NUM_EPISODES = int(os.environ.get('NUM_EPISODES', 7000))
EPSILON_START=1.0
EPSILON_END=0.05
EPSILON_DECAY=5000
TARGET_UPDATE_FREQ=100
INTRINSIC_BETA=0.6
PREDICTION_LOSS_BETA=0.1
PREDICTION_DIM=50
MIN_NUMBER=0 #sonradan ekledim diye hatırkıyorun
MAX_NUMBER=100 #bunu da başarıya kolay ulssın dieydi snrm