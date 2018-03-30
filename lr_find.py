import os
import torch
from fastai.sgdr import *
from fastai.layer_optimizer import *

def save_temp(model):
    os.makedirs("temp", exists_ok=True)
    torch.save(model, "temp/model_lr.t7")

def load_temp():
    return torch.load("temp/model_lr.t7")

def lr_find(self, model, start_lr=1e-5, end_lr=10, wds=None, linear=False):
        """Helps you find an optimal learning rate for a model.
         It uses the technique developed in the 2015 paper
         `Cyclical Learning Rates for Training Neural Networks`, where
         we simply keep increasing the learning rate from a very small value,
         until the loss starts decreasing.
        Args:
            start_lr (float/numpy array) : Passing in a numpy array allows you
                to specify learning rates for a learner's layer_groups
            end_lr (float) : The maximum learning rate to try.
            wds (iterable/float)
        Examples:
            As training moves us closer to the optimal weights for a model,
            the optimal learning rate will be smaller. We can take advantage of
            that knowledge and provide lr_find() with a starting learning rate
            1000x smaller than the model's current learning rate as such:
            >> learn.lr_find(lr/1000)
            >> lrs = np.array([ 1e-4, 1e-3, 1e-2 ])
            >> learn.lr_find(lrs / 1000)
        Notes:
            lr_find() may finish before going through each batch of examples if
            the loss decreases enough.
        .. _Cyclical Learning Rates for Training Neural Networks:
            http://arxiv.org/abs/1506.01186
        """
        #save the temporary model
        save_temp(model)

        #so layer optimizer takes the modules
        #may need to specify one for each part of the model, i.e. one for policy, one for Q
        layer_opt = LayerOptimizer(opt_fn, [model.modules], [start_lr], wds)

        self.sched = LR_Finder(layer_opt, config.EPOCHS, end_lr, linear=linear)
        #do training, during that call the self.sched. stuff
        self.train_memories()

        model = load_temp()

        self.sched.on_train_begin()

        self.sched.on_batch_begin()
        metrics = [loss.data.numpy()[0], valid_loss.data.numpy()[0]]
        self.sched.on_epoch_end(metrics)

        metrics = [loss.data.numpy()[0], valid_loss.data.numpy()[0]]        
        self.sched.on_batch_end(self, metrics)

        self.sched.on_train_end()