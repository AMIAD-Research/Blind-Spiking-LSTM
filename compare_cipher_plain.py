import pandas as pd
import numpy as np
import argparse
import json

def main(config):
    li_y_plain = []
    li_y_cipher = []
    submission_name = config["submission_name"]
    for i in range(1):
        li_y_plain.append(
            pd.read_csv(f"submission_plain_spike/{submission_name}_seed_{i}.tsv",sep="\t")["prediction"]
        )
        li_y_cipher.append(
            pd.read_csv(f"submission_cipher_spike/{submission_name}_seed_{i}.tsv",sep="\t")["prediction"]
        )
    y_plain = np.concat(li_y_plain)
    y_cipher = np.concat(li_y_cipher)
    print(f"Fidelity : {np.sum(y_plain==y_cipher)/y_plain.shape[0]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comparaison Plaintext/Ciphertext inference")
    parser.add_argument("--config",type=str,default="conf_sst2.json")
    args = parser.parse_args()
    config_path = args.config
    with open(config_path) as f:
        config = json.load(f)
    main(config)