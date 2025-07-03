# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import copy

#to implement a switch between SGD and AdamW (changing lr) and the two different type of models

if __name__ == "__main__":
    #Wrtite the code to load the datasets and to run your functions
    # Print the results
    # Device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    #Config
    config = {
        "batch_size_train": 32,
        "batch_size_dev": 128,
        "batch_size_test": 128,
        "hid_size": 200,
        "emb_size": 200,
        "lr": 2, # 2 with SGD, 0.001 with adamw
        "clip": 5, # normalize the gradient if >5
        "n_epochs": 100,
        "patience": 3,
        "mode":1 #1 for LSTM, 2 for LSTM with dropout, 3 for LSTM with AdamW
    }
    
    #input the mode you want to use
    mode = int(input("Enter the mode you want to use (1 for LSTM, 2 for LSTM with dropout, 3 for LSTM with AdamW): "))
    if mode not in [1, 2, 3]:
        raise ValueError("Invalid mode. Please enter 1, 2 or 3.")
    # Load the dataset
    train_raw = read_file("exam/studentID_name_surname/LM/dataset/ptb.train.txt")
    dev_raw = read_file("exam/studentID_name_surname/LM/dataset/ptb.valid.txt")
    test_raw = read_file("exam/studentID_name_surname/LM/dataset/ptb.test.txt")
    # Create the vocab
    vocab = get_vocab(train_raw, special_tokens=["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    # Create the dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=64, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    vocab_len = len(lang.word2id)
    # Create the model
    if mode == 1:
        config["mode"] = 1
        model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    elif mode == 2:
        config["mode"] = 2
        model = LM_LSTM_dropout(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    elif mode == 3:
        config["mode"] = 3
        model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    else:
        print("Invalid mode, using default LSTM model")
        config["mode"] = 1
        model = LM_LSTM(config["emb_size"], config["hid_size"], vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
    
    model.apply(init_weights)

    if config["mode"] == 3:
        #AdamW
        optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    else:
        #SGD
        optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Training loop
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, config["n_epochs"]))
        
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, config["clip"])
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(np.asarray(loss_dev).mean())
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl:  # the lower, the better
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1

            if patience <= 0:  # Early stopping with patience
                break  # Not nice but it keeps the code clean

    best_model.to(DEVICE)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)
    
    #store the model
    path = 'model_bin/model_name.pt'
    torch.save(model.state_dict(), path)