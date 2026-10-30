import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def visualize_heads(writer, att_map, cols, step, num):
    to_shows = []
    batch_num = att_map.shape[0]
    head_num = att_map.shape[1]
    # att_map = att_map.squeeze()
    for i in range(batch_num):
        for j in range(head_num):
            to_shows.append((att_map[i][j], f'Batch {i} Head {j}'))
        average_att_map = att_map[i].mean(axis=0)
        to_shows.append((average_att_map, f'Batch {i} Head Average'))

    rows = (len(to_shows)-1) // cols + 1
    it = iter(to_shows)
    fig, axs = plt.subplots(rows, cols, figsize=(rows*8.5, cols*2))
    for i in range(rows):
        for j in range(cols):
            try:
                image, title = next(it)
            except StopIteration:
                image = np.zeros_like(to_shows[0][0])
                title = 'pad'
            axs[i, j].imshow(image)
            axs[i, j].set_title(title)
            axs[i, j].set_yticks([])
            axs[i, j].set_xticks([])

    writer.add_figure("attention_{}".format(num), fig, step)
    # plt.show()