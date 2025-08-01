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

 # IAS -> F1: 0.9262, Intent Accuracy: 0.9160
 # Model with bidirectionality -> F1: 0.9459, Intent Accuracy: 0.9507
 # Model with dropout and bidirectionality -> F1: 0.9343, Intent Accuracy: 0.9384

if __name__ == "__main__":
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" # Used to report errors on CUDA side
    PAD_TOKEN = 0
    
    
    current_dir = os.path.dirname(os.path.realpath(__file__))
    tmp_train_raw = load_data(os.path.join(current_dir, 'dataset', 'train.json'))
    test_raw = load_data(os.path.join(current_dir, 'dataset', 'test.json'))
    
    train_raw, dev_raw = dev_set(tmp_train_raw)
    
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
    mode = int(input("Enter 1 for training, otherwise enter 2 for testing: "))
    if mode == 2:
        model_choice = int(input("Enter the model you want to test (1 for ModelIAS, 2 for bidirectional ModelIAS, 3 for ModelIAS with dropout): "))

        if model_choice == 1:
            model_name = "model_ias.pt"
        elif model_choice == 2:
            model_name = "model_ias_bidirect.pt"
        elif model_choice == 3:
            model_name = "model_ias_dropout.pt"

        checkpoint = torch.load(os.path.join("bin", model_name), map_location=device)

        # recover the language object
        lang = Lang(words=[], intents=[], slots=[], cutoff=0)
        lang.word2id = checkpoint["w2id"]
        lang.slot2id = checkpoint["slot2id"]
        lang.intent2id = checkpoint["intent2id"]
        lang.id2word = {v:k for k,v in lang.word2id.items()}
        lang.id2slot = {v:k for k,v in lang.slot2id.items()}
        lang.id2intent = {v:k for k,v in lang.intent2id.items()}

        # Ora carica i dati
        current_dir = os.path.dirname(os.path.realpath(__file__))
        test_raw = load_data(os.path.join(current_dir, 'dataset', 'test.json'))
        test_dataset = IntentsAndSlots(test_raw, lang)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size_test'], collate_fn=collate_fn)

        # Instanzia e carica il modello
        out_slot  = len(lang.slot2id)
        out_int   = len(lang.intent2id)
        vocab_len = len(lang.word2id)
        bidirect_flag = (model_choice >= 2)
        dropout_flag = (model_choice == 3)

        model = build_model(config, out_slot, out_int, vocab_len, PAD_TOKEN, bidirect_flag, dropout_flag)
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        model.eval()

        # Evaluation
        results_test, intent_test, _ = eval_loop(test_loader,
                                                nn.CrossEntropyLoss(ignore_index=PAD_TOKEN),
                                                nn.CrossEntropyLoss(),
                                                model, lang)

        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])
        exit(0)

    elif mode != 1 or mode != 2:
        print("Invalid choice. Defaulting to training mode.")
    else:
        mode_choice = int(input("Enter the mode you want to use (1 for ModelIAS, 2 for bidirectional ModelIAS, 3 for ModelIAS with dropout): "))

        if mode_choice == 1:
            bidirect_flag = False
            dropout_flag = False
            model_name = "model_ias.pt"
        elif mode_choice == 2:
            bidirect_flag = True
            dropout_flag = False
            model_name = "model_ias_bidirect.pt"
        elif mode_choice == 3:
            bidirect_flag = True
            dropout_flag = True
            model_name = "model_ias_dropout.pt"
        else:
            print("Invalid choice. Defaulting to ModelIAS without bidirectional and dropout.")
            bid_flag = False
            dropout_flag = False
        
        model = build_model(config, out_slot, out_int, vocab_len, PAD_TOKEN, bidirect_flag, dropout_flag)
        
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
                    
                    save_model(model, optimizer, lang.word2id, lang.slot2id, lang.intent2id, x, "bin", model_name)
                    print("New best model saved with F1: {:.4f}".format(f1))
                    
                    patience = 3
                else:
                    patience -= 1
                if patience <= 0: # Early stopping with patience
                    break # Not nice but it keeps the code clean

        results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)    
        print('Slot F1: ', results_test['total']['f'])
        print('Intent Accuracy:', intent_test['accuracy'])
        
    