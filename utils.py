import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from collections import OrderedDict


class Utils:

    def __init__(self):
        pass


    def plot_history(self, history, artifact_path, file_name,
                     right_limit, top_limit):
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.plot(history['train_acc'], label='Training Accuracy')
        plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Loss and Accuracy Chart')
        plt.xlabel('Number of epochs')
        plt.ylabel('Loss/Acc')
        plt.xlim(right=right_limit)
        plt.ylim(top=top_limit)
        plt.yticks(np.arange(0, 1.1, 0.1))
        plt.legend(loc='center right')
        plt.savefig(os.path.join(artifact_path, file_name))



    def save_checkpoint(self, history_dict, artifact_path, check_point_name):
        torch.save(history_dict, os.path.join(artifact_path, check_point_name))