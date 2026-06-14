from datasets import load_dataset


def load_data():

    print("Loading Dataset...")

    dataset = load_dataset(
        "Tobi-Bueck/customer-support-tickets"
    )

    data = dataset["train"]

    print(
        f"Loaded {len(data)} samples"
    )

    return data


if __name__ == "__main__":

    data = load_data()

    print("\nDataset Loaded Successfully")

    print(
        "\nFirst Record:\n",
        data[0]
    )