from learnedcache.loading import transform_logs_to_csvs, read_csvs_to_dataframe
from sklearn.model_selection import train_test_split

def main():
    transform_logs_to_csvs('data/fileserver/*.log')
    df = read_csvs_to_dataframe('data/fileserver/*access.csv')
    print("Combined DataFrame shape:", df.shape)

if __name__ == "__main__":
    main()
