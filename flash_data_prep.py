import pandas as pd
import numpy as np
import glob
import os
import dataprep



def build_port_TDL(row, min_delta=1e-5):
    to_return = []

    measures = np.array(row['udps.protocol_header_fields_enh'][:, 1])
    # sizes = (measures - measures.min()) / (measures.max() - measures.min())
    sizes = measures #  (maybe I can change all zeroes to 1. so it will be different than no data)
    directions = [0]*len(measures)
    for el_i, el in enumerate(row['udps.protocol_header_fields_enh'][:, 0]): # enumerate(np.array(hlp_protocol_header_fields_enh[:,0])): # enumerate(row['udps.protocol_header_fields_enh'][:, 0]):
        if el != 0:
            directions[el_i] = 1
            # normalized_measures[el_i] *= -1
    measures = np.array(row['udps.protocol_header_fields_enh'][:, 2])
    measures = measures / 1000
    times = [0] * len(measures)
    for i in range(1, len(times)):
        times[i] = times[i-1] + max(measures[i], min_delta)
    # if measures.max() != measures.min():    # we want to avoid 0/0
    #     times = (measures - measures.min()) / (measures.max() - measures.min())

    for item in zip(times, directions, sizes):
        to_return.append(list(item))

    return to_return


def add_to_df(file_path):
    df = pd.read_csv(file_path)
    df = df[df["new_application_types"]!="other"]
    df['filename'] = file_path.split('/')[-1]
    df['packet_number'] = df['udps.protocol_header_fields_enh'].str.count('\n')
    # this raw convert the nfstream plugin output from string to list of lists - for now it is crashing
    df['udps.protocol_header_fields_enh'] = df['udps.protocol_header_fields_enh'].apply(lambda x: np.array([list(map(int, row.strip().strip('[]').split())) for row in x.strip('[]').split('\n') if row.strip()]))
    df["tdl"] = df.apply(build_port_TDL, axis=1)
    return df


def extract_flowpic_data(df, flowpic_dim=32, flowpic_block_duration=15):
    df_tdls = df['tdl'].apply(lambda x: pd.DataFrame(x, columns=['timetofirst', 'pkts_dir', 'pkts_size']))
    np_flowpics = []
    for idx, tdl in enumerate(df_tdls):
        tdl['app'] = df.iloc[idx]["new_application_types"]
        np_flowpic = tdl_to_np_flowpic(tdl, flowpic_dim=flowpic_dim, flowpic_block_duration=flowpic_block_duration)
        np_flowpics.append(np_flowpic)
    return np_flowpics


def tdl_to_np_flowpic(tdl_df, flowpic_dim=32, flowpic_block_duration=15):
    dset = dataprep.FlowpicDataset(
        data=tdl_df,
        timetofirst_colname="timetofirst",
        pkts_size_colname="pkts_size",
        pkts_dir_colname="pkts_dir",
        target_colname="app",
        flowpic_dim=flowpic_dim,
        flowpic_block_duration=flowpic_block_duration,
        n_workers=1
    )

    # fetch the flowpic representation
    flowpic, label = dset[0]

    # flattening the representation
    # to remove zero values (used for finding
    # min values)
    flowpic = flowpic.numpy().squeeze()
    flattened = flowpic.flatten()
    flattened = flattened[flattened > 0]
    return flattened

    


def main():
    
    data_dir = "/home/jony/git/GAN-N-Net/datasets/new_raw_data/site5_1500_packets_al_to_share/csvs/"
    dest_dir = os.path.join(os.path.dirname(data_dir), os.path.basename(data_dir) + "_np")
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    files = glob.glob(data_dir + "*.csv")
    for file in files:
        df = add_to_df(file)
        np_flowpics = extract_flowpic_data(df)

        np.save(os.path.join(dest_dir, os.path.basename(file).replace(".csv", ".npy")), np_flowpics)


if __name__ == "__main__":
    main()