from utils import *
from functions import *

#parameters for the model
config = {
    "batch_size_train": 128, #original 128
    "batch_size_dev": 64, #original 64
    "batch_size_test": 64, #original 64
    "lr": 0.005,
    "hid_size": 200,
    "emb_size": 300,
    "dropout": 0.3,
    "clip": 5,
    "n_epochs": 100,
    "patience_init": 3,
}


if __name__ == "__main__":
    # Device
    device = 'cuda:0' # cuda:0 means we are using the GPU with id 0, if you have multiple GPU
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    PAD_TOKEN = 0
    
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, 'dataset', 'train.json'))
    test_raw = load_data(os.path.join(current_dir, 'dataset', 'test.json'))
    
    train_raw, dev_raw = get_dev(tmp_train_raw)
    
    words = sum([x['utterance'].split() for x in train_raw], []) # No set() since we want to compute the cutoff
    corpus = train_raw + dev_raw + test_raw # We do not wat unk labels, however this depends on the research purpose
    slots = set(sum([line['slots'].split() for line in corpus],[]))
    intents = set([line['intent'] for line in corpus])

    lang = Lang(words, intents, slots, cutoff=0)
    
    # Create our datasets
    train_dataset = IntentsAndSlots(train_raw, lang)
    dev_dataset = IntentsAndSlots(dev_raw, lang)
    test_dataset = IntentsAndSlots(test_raw, lang)

    # Dataloader instantiations
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size_train'], collate_fn=collate_fn,  shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=config['batch_size_dev'], collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], collate_fn=collate_fn)


    out_slot = len(lang.slot2id)
    out_int = len(lang.intent2id)
    vocab_len = len(lang.word2id)

    # Create the model
        
    #choose the mode
    mode_choice = int(input("Enter the mode you want to use (1 for ModelIAS, 2 for bidirectional ModelIAS, 3 for ModelIAS with dropout): "))
    if mode_choice == 1:
        model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], vocab_len, pad_index=PAD_TOKEN).to(device)
    elif mode_choice == 2:
        model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], vocab_len, pad_index=PAD_TOKEN, apply_bidirectional=True).to(device)
    elif mode_choice == 3:
        model = ModelIAS(config["hid_size"], out_slot, out_int, config["emb_size"], vocab_len, pad_index=PAD_TOKEN, apply_bidirectional=True, apply_dropout=True, dropout=config["dropout"]).to(device)
    else:
        print("Invalid mode, using default ModelIAS model")

    model.apply(init_weights)

    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss() # Because we do not have the pad token


    losses_train = []
    losses_dev = []
    sampled_epochs = []
    best_f1 = 0
    for x in tqdm(range(1,config["n_epochs"])):
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=config["clip"])
        if x % 5 == 0: # We check the performance every 5 epochs
            sampled_epochs.append(x)
            losses_train.append(np.asarray(loss).mean())
            results_dev, intent_res, loss_dev = eval_loop(dev_loader, criterion_slots, criterion_intents, model, lang)
            losses_dev.append(np.asarray(loss_dev).mean())
            
            f1 = results_dev['total']['f']
            # For decreasing the patience you can also use the average between slot f1 and intent accuracy
            if f1 > best_f1:
                best_f1 = f1
                # Here you should save the model
                torch.save(model.state_dict(), 'best_model.pth')
                print(f"New best F1: {f1}, saving model.")
                
                patience = 3
            else:
                patience -= 1
            if patience <= 0: # Early stopping with patience
                break # Not nice but it keeps the code clean

    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
    print('Slot F1: ', results_test['total']['f'])
    print('Intent Accuracy:', intent_test['accuracy'])
