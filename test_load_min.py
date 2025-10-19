from data.DataLoader import AmazonDataset

# Test loading the min train file
try:
    dataset = AmazonDataset('./data/processed/train.mini.csv', max_num_reviews=8, is_train=True, refs_path=None, vocab=None, max_len_rev=None, preprocess=False)
    print(f"Dataset loaded successfully. Number of batches: {len(dataset)}")
    print("Sample batch indexes:", dataset.batch_indexes[:5])
    print("Sample reviews (first product):", len(dataset.src_reviews[:dataset.batch_indexes[1]]))
except Exception as e:
    print(f"Error loading dataset: {e}")
