import pandas as pd
from collections import Counter
import argparse


def main(args):
    # Initialize dictionaries to store data and a set to store unique IDs
    data_maps = {file_path: {} for file_path in args.csv_files}
    unique_ids = []

    is_subclass = args.is_subclass

    unseen_label = 3
    if is_subclass:
        unseen_label = 87

    for file_path in args.csv_files:
        # Read CSV file
        df = pd.read_csv(file_path)

        # Store unique IDs
        if len(unique_ids) == 0:
            unique_ids = list(df['ID'])

        # Convert DataFrame columns to a dictionary and store in the corresponding map
        data_maps[file_path] = dict(zip(df['ID'], df['Target']))

    output = {"ID": [], "Target": []}

    for id in unique_ids:
        values = []

        for file_path, data_map in data_maps.items():
            value = data_map[id]
            values.append(value)

        integer_counter = Counter(values)
        most_common_integer, count = integer_counter.most_common(1)[0]
        # If 2 or more people predict the same result for this image
        output['ID'].append(id)
        if count > 1:
            output['Target'].append(most_common_integer)
        else:
            # If all 3 people predict different label
            output['Target'].append(unseen_label)

    # print("List of Unique IDs:")
    # print(unique_ids)

    output = pd.DataFrame(data=output)
    output.to_csv('ensemble_prediction.csv', index=False)
    print('ensemble done!')

if __name__ == "__main__":
    # Create an argument parser
    parser = argparse.ArgumentParser(description="Process CSV files.")

    ensemble_data_dir = "ensemble_test_data/"
    parser.add_argument("csv_files", nargs="*", default=[ensemble_data_dir + "test_predictions_sub1.csv", ensemble_data_dir + "test_predictions_sub2.csv", ensemble_data_dir + "test_predictions_sub3.csv"], help="List of CSV file paths")
    parser.add_argument("--is_subclass", action="store_true", default=True,
                        help="Flag indicating whether the files represent subclass predictions")

    # Parse the command-line arguments
    args = parser.parse_args()

    # Call the main function with the parsed arguments
    main(args)
