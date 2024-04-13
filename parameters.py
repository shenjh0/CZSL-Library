import argparse

parser = argparse.ArgumentParser()


YML_PATH = {
    "mit-states": './configs/mit-states.yml',
    "ut-zappos": './configs/ut-zappos.yml',
    "cgqa": './configs/cgqa.yml'
}


#model config
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/CompCos.yml')
parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/PROLT.yml') ## TODO: fix it
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/ivr.yml')
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/scen.yml')
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/CANet.yml')
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/CoT.yml')
# parser.add_argument("--cfg", help="cofig file path", type=str, default='configs/mit-states.yml')

parser.add_argument("--dataset", help="name of the dataset", type=str, default='mit-states')
parser.add_argument("--dataset_path", help="name of the dataset", type=str, default='data/mit-states')
parser.add_argument("--word_embedding_root", help="name of the dataset", type=str, default='data')

parser.add_argument("--lr", help="learning rate", type=float, default=5e-05)
parser.add_argument("--weight_decay", help="weight decay", type=float, default=1e-05)
parser.add_argument("--clip_model", help="clip model type", type=str, default="ViT-L/14")
parser.add_argument("--epochs", help="number of epochs", default=20, type=int)
parser.add_argument("--epoch_start", help="start epoch", default=0, type=int)
parser.add_argument("--train_batch_size", help="train batch size", default=48, type=int)
parser.add_argument("--eval_batch_size", help="eval batch size", default=16, type=int)
parser.add_argument("--fusion", default="BiFusion", help="cross modal fusion method, choices = [BiFusion, txt2img, img2txt, NoFusion, DeCom]",)
parser.add_argument("--context_length", help="sets the context length of the clip model", default=8, type=int)
parser.add_argument("--attr_dropout", help="add dropout to attributes", type=float, default=0.3)
parser.add_argument("--save_path", help="save path", type=str)
parser.add_argument("--save_every_n", default=5, type=int, help="saves the model every n epochs")
parser.add_argument("--save_model", help="indicate if you want to save the model state dict()", action="store_true")
parser.add_argument("--load_model", default=None, help="load the trained model")
parser.add_argument("--seed", help="seed value", default=0, type=int)
parser.add_argument("--gradient_accumulation_steps", help="number of gradient accumulation steps", default=1, type=int)

parser.add_argument("--open_world", help="evaluate on open world setup", default= False)
parser.add_argument("--bias", help="eval bias", type=float, default=1e3)
parser.add_argument("--topk", help="eval topk", type=int, default=1)
parser.add_argument("--text_encoder_batch_size", help="batch size of the text encoder", default=16, type=int)
parser.add_argument('--threshold', type=float, help="optional threshold")
parser.add_argument('--threshold_trials', type=int, default=50, help="how many threshold values to try")
parser.add_argument('--imagenet', type=bool, default=False, )