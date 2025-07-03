# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file
from functions import *
from utils import *
from model import *
import copy

if __name__ == "__main__":
    # Device
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    # Config
    config = {
        "batch_size_train": 32,
        "batch_size_dev": 128,
        "batch_size_test": 128,
        "hid_size": 256,  
        "emb_size": 256,  
        "lr": 1.5,        # Ottimizzato per SGD
        "clip": 5,
        "n_epochs": 100,
        "patience": 5,    
        "emb_dropout": 0.4,  # Solo per variational dropout
        "out_dropout": 0.4   # Solo per variational dropout
    }
    
    # Input the model type
    mode = input("Enter the model type (1 for LSTM with weight tying, 2 for LSTM with variational dropout, 3 for LSTM with Non-monotonically Triggered AvSGD): ")
    if mode not in ["1", "2", "3"]:
        raise ValueError("Invalid model type. Please enter 1, 2 or 3.")
    
    # Load the dataset
    train_raw = read_file("exam/258237-Giampietro-Andrea/LM/dataset/ptb.train.txt")
    dev_raw = read_file("exam/258237-Giampietro-Andrea/LM/dataset/ptb.valid.txt")
    test_raw = read_file("exam/258237-Giampietro-Andrea/LM/dataset/ptb.test.txt")
    
    # Create the vocab
    vocab = get_vocab(train_raw, special_tokens=["<pad>", "<eos>"])
    lang = Lang(train_raw, ["<pad>", "<eos>"])
    
    # Create the dataset
    train_dataset = PennTreeBank(train_raw, lang)
    dev_dataset = PennTreeBank(dev_raw, lang)
    test_dataset = PennTreeBank(test_raw, lang)
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size_train"], 
                             collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]), shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config["batch_size_dev"], 
                           collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size_test"], 
                            collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
    
    vocab_len = len(lang.word2id)

    # Model selection based on Part 1.B requirements (CUMULATIVE)
    if mode == "1":
        # LSTM + Weight Tying
        model = LM_LSTM_weight_tying(
            emb_size=config["emb_size"], 
            hidden_size=config["hid_size"], 
            output_size=vocab_len, 
            pad_index=lang.word2id["<pad>"]
        )
        print("Mode 1: LSTM with Weight Tying")
        
    elif mode == "2":
        # LSTM + Weight Tying + Variational Dropout
        model = LM_LSTM_variational_dropout(
            emb_size=config["emb_size"], 
            hidden_size=config["hid_size"], 
            output_size=vocab_len, 
            pad_index=lang.word2id["<pad>"],
            emb_dropout=config["emb_dropout"],
            out_dropout=config["out_dropout"]
        )
        print("Mode 2: LSTM with Weight Tying + Variational Dropout")
        
    elif mode == "3":
        # LSTM + Weight Tying + Variational Dropout + NT-AvSGD
        model = LM_LSTM_nt_avsgd(
            emb_size=config["emb_size"], 
            hidden_size=config["hid_size"], 
            output_size=vocab_len, 
            pad_index=lang.word2id["<pad>"],
            emb_dropout=config["emb_dropout"],
            out_dropout=config["out_dropout"]
        )
        print("Mode 3: LSTM with Weight Tying + Variational Dropout + NT-AvSGD")
    
    model.to(DEVICE)
    model.apply(init_weights)

    # Optimizer
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])
    
    # Loss functions
    criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

    # Initialize NT-AvSGD tracker for mode 3
    nt_avsgd_tracker = None
    if mode == "3":
        nt_avsgd_tracker = NTAvSGDTracker(model, dev_loader, criterion_eval, non_monotone_interval=5)

    # Training loop
    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    patience = config["patience"]
    
    pbar = tqdm(range(1, config["n_epochs"] + 1))
        
    for epoch in pbar:
        # Training
        if mode == "1" or mode == "2":
            loss = train_loop(train_loader, optimizer, criterion_train, model, config["clip"])
        elif mode == "3":
            loss = train_with_AvSGD(train_loader, optimizer, criterion_train, model, config["clip"])
        
        # Validation every epoch
        sampled_epochs.append(epoch)
        losses_train.append(loss)
        
        if mode == "3":
            # Use NT-AvSGD validation check
            ppl_dev = nt_avsgd_tracker.validation_check(epoch)
            losses_dev.append(ppl_dev)
        else:
            # Standard validation
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(loss_dev)
        
        pbar.set_description("PPL: %f" % ppl_dev)
        
        # Best model tracking
        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = config["patience"]
        else:
            patience -= 1

        if patience <= 0:
            break

    # Finalize NT-AvSGD if used
    if mode == "3" and nt_avsgd_tracker:
        print("Applying NT-AvSGD weight averaging...")
        nt_avsgd_tracker.finalize()
        # Re-evaluate after averaging
        final_ppl, _ = eval_loop(test_loader, criterion_eval, model)
        print('Test ppl after NT-AvSGD averaging: ', final_ppl)
    else:
        # Use best model for final evaluation
        best_model.to(DEVICE)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print('Test ppl: ', final_ppl)
    
    # Store the model
    path = f'model_bin/model_mode_{mode}.pt'
    if mode == "3":
        torch.save(model.state_dict(), path)  # Save the averaged model
    else:
        torch.save(best_model.state_dict(), path)
    
    print(f"Best validation PPL: {best_ppl}")
    print(f"Final test PPL: {final_ppl}")
    
    # Print training summary
    print(f"\nTraining Summary:")
    print(f"Mode: {mode}")
    print(f"Hidden size: {config['hid_size']}")
    print(f"Embedding size: {config['emb_size']}")
    print(f"Learning rate: {config['lr']}")
    if mode == "2":
        print(f"Dropout rates: emb={config['emb_dropout']}, out={config['out_dropout']}")
    print(f"Epochs trained: {epoch}")
    print(f"Best validation PPL: {best_ppl:.2f}")
    print(f"Final test PPL: {final_ppl:.2f}")