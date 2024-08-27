import pandas as pd
import os 
from scipy.stats import zscore

def realism_handling(data):
    """
    Processes a dictionary of dictionaries or a list of such dictionaries to
    extract realism scores, compute Z-scores, and calculate averaged Z-scores.

    Parameters:
    data (dict or list): A dictionary where each value is another dictionary
                         containing various metadata, or a list of such dictionaries.

    Returns:
    pd.DataFrame: A DataFrame where the index is the filename, and the columns
                  contain the realism scores for each JSONL file, normalized Z-scores,
                  and averaged Z-scores.
    """

    def extract_realism_scores(single_data):
        """Helper function to extract realism scores from a single dict of dicts."""
        realism_scores = {}

        for key, value in single_data.items():
            # Extract the local_path (filename) and realism_score
            filename = value.get('local_path')
            realism_score = value.get('realism_score')

            # Use the filename as the key and realism score as the value
            if filename is not None:
                realism_scores[os.path.basename(filename)] = realism_score

        return realism_scores

    # Handle multiple JSONL files (list of dicts)
    if isinstance(data, list):
        combined_realism_scores = {}

        for idx, single_data in enumerate(data):
            # Extract realism scores for each JSONL file and store them with a unique column name
            realism_scores = extract_realism_scores(single_data)
            combined_realism_scores[f'Realism Score {idx+1}'] = pd.Series(realism_scores)

        # Create a DataFrame with columns for each JSONL file's realism scores
        df = pd.DataFrame(combined_realism_scores)

        # Compute Z-scores for each JSONL file
        for idx in range(len(data)):
            df[f'Z-Score {idx+1}'] = zscore(df[f'Realism Score {idx+1}'], nan_policy='omit')

        # Calculate the averaged Z-score across all JSONL files
        z_score_columns = [f'Z-Score {idx+1}' for idx in range(len(data))]
        df['Averaged Z-Score'] = df[z_score_columns].mean(axis=1)

        # Calculate the averaged Z-score across all JSONL files
        realism_score_columns = [f'Realism Score {idx+1}' for idx in range(len(data))]
        df['Averaged Realism Score'] = df[realism_score_columns].mean(axis=1)


    # Handle a single JSONL file (dict of dicts)
    elif isinstance(data, dict):
        realism_scores = extract_realism_scores(data)
        df = pd.DataFrame.from_dict(realism_scores, orient='index', columns=['Realism Score'])

        # Compute the Z-score for the single JSONL file
        df['Z-Score'] = zscore(df['Realism Score'], nan_policy='omit')

    else:
        raise ValueError("Input data must be either a dict or a list of dicts.")

    return df

# Example usage:
if __name__ == "__main__":
    data = [
        {
            0: {'user_id': '4e26daae-530d-430b-81be-107704de6a9e', 'image_path': 'https://example.com/image1.png', 'category': 'real_calc', 'is_real': False, 'realism_score': 2.5, 'image_duration': 4.76, 'index': 39, 'timestamp': '2024-08-19T09:26:32.421254', 'local_path': './data/real/image1.png'},
            31: {'user_id': '4e26daae-530d-430b-81be-107704de6a9e', 'image_path': 'https://example.com/image2.png', 'category': 'synth_diff_calc', 'is_real': False, 'realism_score': 1.0, 'image_duration': 25.58, 'index': 1, 'timestamp': '2024-08-19T08:37:34.299025', 'local_path': './data/synthetic/image2.png'}
        },
        {
            0: {'user_id': '4e26daae-530d-430b-81be-107704de6a9e', 'image_path': 'https://example.com/image1.png', 'category': 'real_calc', 'is_real': False, 'realism_score': 3.0, 'image_duration': 4.76, 'index': 39, 'timestamp': '2024-08-19T09:26:32.421254', 'local_path': './data/real/image1.png'},
            31: {'user_id': '4e26daae-530d-430b-81be-107704de6a9e', 'image_path': 'https://example.com/image2.png', 'category': 'synth_diff_calc', 'is_real': False, 'realism_score': 0.5, 'image_duration': 25.58, 'index': 1, 'timestamp': '2024-08-19T08:37:34.299025', 'local_path': './data/synthetic/image2.png'}
        }
    ]

    df = realism_handling(data)
    print(df)

