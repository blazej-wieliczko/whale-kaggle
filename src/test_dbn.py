network = dbn.DBN([-1, 500, 500, -1], momentum=0.9, learn_rates=0.3, learn_rate_minimums=0.01,learn_rate_decays=0.996,
            epochs=400,minibatch_size=100, fine_tune_callback=evaluate_validation, epochs_pretrain=[100, 60, 60, 60],
            dropouts=[0.1, 0.5, 0.5, 0],momentum_pretrain=0.9, learn_rates_pretrain=[0.001, 0.01, 0.01, 0.01],
            verbose=1)
