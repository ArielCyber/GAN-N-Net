import pandas as pd
import matplotlib.pyplot as plt
import os


# Load the data from CSV files

def create_plot_comarison_graph(csv_input_1, csv_input_2, output_path, test_1_label=None, test_2_label=None):
    """
    Generates a comparison graph from two CSV files with test results.

        Parameters:
        csv_input_1 (str): The file path to the first CSV input.
        csv_input_2 (str): The file path to the second CSV input.
        output_path (str): The file path where to save the output graph.
        test_1_label (str, optional): Label for the first test data. Defaults
                                      to the basename of csv_input_1 if None.
        test_2_label (str, optional): Label for the second test data. Defaults
                                      to the basename of csv_input_2 if None.

        Returns:
        None: The function saves the plot to the specified output_path and
              prints the last row of the processed dataframes to the console.
    """

    df_input_1 = pd.read_csv(csv_input_1)
    df_input_2 = pd.read_csv(csv_input_2)
    if not test_1_label:
        test_1_label = os.path.basename(csv_input_1)
    if not test_2_label:
        test_2_label = os.path.basename(csv_input_2)
    # Create a list of epoch numbers from 1 to 200
    epochs = range(1, 201)

    # Create a figure with 3 subplots
    # fig, axs = plt.subplots(1, 4, figsize=(15, 5))
    fig, axs = plt.subplots(4, 1, figsize=(5, 10))


    # Plot the data on the accuracy subplot
    axs[0].plot(epochs, df_input_1['val_test_accuracy'], label=test_1_label)
    axs[0].plot(epochs, df_input_2['val_test_accuracy'], label=test_2_label)
    # axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Accuracy')
    # axs[0].set_title('Accuracy')
    # axs[0].legend()

    # Plot the data on the precision subplot
    axs[1].plot(epochs, df_input_1['val_test precision'], label=test_1_label)
    axs[1].plot(epochs, df_input_2['val_test precision'], label=test_2_label)
    # axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Precision')
    # axs[1].set_title('Precision')
    # axs[1].legend()

    # Plot the data on the recall subplot
    axs[2].plot(epochs, df_input_1['val_test recall'], label=test_1_label)
    axs[2].plot(epochs, df_input_2['val_test recall'], label=test_2_label)
    # axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('Recall')
    # axs[2].set_title('Recall')
    # axs[2].legend()


    # Plot the data on the F1-score subplot
    df_input_1["f1_score"] = (2 * (df_input_1['val_test precision'] * df_input_1['val_test recall'])) / (df_input_1['val_test precision'] + df_input_1['val_test recall'])
    df_input_2["f1_score"] = (2 * (df_input_2['val_test precision'] * df_input_2['val_test recall'])) / (df_input_2['val_test precision'] + df_input_2['val_test recall'])

    axs[3].plot(epochs, df_input_1['f1_score'], label=test_1_label)
    axs[3].plot(epochs, df_input_2['f1_score'], label=test_2_label)
    axs[3].set_xlabel('Epoch')
    axs[3].set_ylabel('F1-Score')
    # axs[3].set_title('F1-Score')
    # axs[3].legend()

    # Create a single legend for all subplots and place it at the top
    # handles, labels = axs[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper left')#, bbox_to_anchor=(0.5, 1.1), ncol=2)

    # Adjust layout to prevent overlapping
    # fig.tight_layout()

    # Save the figure to a file
    fig.savefig(output_path)
    print(df_input_1.tail(1))
    print()
    print(df_input_2.tail(1))

    # Display the figure
    # plt.show()


if __name__ == '__main__':
    # QUIC Davis - quic text
    # QUIC Paris-Est Créteil - quic pcap
    dataset_name = "mirage22"

    # create folder plost if not exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    if dataset_name == "QUIC Davis":  # quic text
        gan_path = "console_output/training_quic_text_gan.csv"
        no_gan_path = "console_output/training_quic_text_nogan.csv"
        output_plot_path = 'plots/vertical_quic_text_gan_vs_nogan.png'
        # plot_label_1 = "QUIC Davis with GAN-N-Net"
        # plot_label_2 = "QUIC Davis"

    elif dataset_name == "QUIC Paris":  # quic pcap
        gan_path = "console_output/training_quic_pcap_gan.csv"
        no_gan_path = "console_output/training_quic_pcap_nogan.csv"
        output_plot_path = "plots/vertical_chen_score_quic_pcap_gan_vs_nogan.png"

        # plot_label_1 = "QUIC Paris-Est Créteil with GAN-N-Net"
        # plot_label_2 = "QUIC Paris-Est Créteil"

    elif dataset_name == "mirage22":  # quic pcap
        gan_path = "console_output/training_mirage22_100_training_200_epochs.csv"
        no_gan_path = "console_output/training_mirage22_100_training_200_epochs_no_gan.csv"
        output_plot_path = "plots/vertical_mirage22_100_gan_vs_nogan.png"

        # plot_label_1 = "Mirage 22 with GAN-N-Net"
        # plot_label_2 = "Mirage 22"

    plot_label_1 = "With GAN-N-Net"
    plot_label_2 = "Basic Classification"
    create_plot_comarison_graph(gan_path, no_gan_path, output_plot_path, test_1_label=plot_label_1, test_2_label=plot_label_2)