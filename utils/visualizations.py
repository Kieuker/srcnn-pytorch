import matplotlib.pyplot as plt
import os

def display_set5_sr(image_list, export_dir=None, export_filename=None, is_grayscale=False):
    # Setting images
    fig, axs = plt.subplots(5, 3, figsize=(15, 25))

    axs[0][0].set_title("Low Resolution (LR)")
    axs[0][1].set_title("Super Resolution (SR)")
    axs[0][2].set_title("Ground Truth (GT)")

    for row in range(5):
        if is_grayscale:
            axs[row][0].imshow(image_list[row][0], cmap='gray')
            axs[row][1].imshow(image_list[row][1], cmap='gray')
            axs[row][2].imshow(image_list[row][2], cmap='gray')
        else:
            axs[row][0].imshow(image_list[row][0])
            axs[row][1].imshow(image_list[row][1])
            axs[row][2].imshow(image_list[row][2])
        
        axs[row][0].axis('off')
        axs[row][1].axis('off')
        axs[row][2].axis('off')
    
    plt.tight_layout()

    # Export plot into file
    if export_dir and export_filename:
        export_path = os.path.join(export_dir, export_filename)
        plt.savefig(export_path)

    plt.show()
    plt.close()
