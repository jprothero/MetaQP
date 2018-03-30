def save_temp(model):
    os.makedirs("temp", exists_ok=True)
    torch.save(model, "temp/model_lr.t7")

def load_temp():
    return torch.load("temp/model_lr.t7")

def test_learning_rate(model, optim, starting_lr):
    save_temp(model)
    temp_model = load_temp()
    
    loss = 1e8
    last_loss = 1e9

    lr = starting_lr
    last_lr = lr

    e = 1
    while loss < last_loss:
        last_loss = loss
        last_lr = lr

        LR = CyclicLR(min_lr = lr/4, max_lr=lr, step=4*config.TRAINING_BATCH_SIZE)

        clr = LR.get_rate(epoch=e, 1)
        adjust_learning_rate(optim, clr)
        print("CLR lr: {}".format(rate))

        lr *= 3

    return last_lr/3