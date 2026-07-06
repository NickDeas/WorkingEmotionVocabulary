# Make Scripts Executable
chmod +x code/*

# Download MASIVE
gdown 1MxvcL7M3iGa4-j_hznMTBLXe9ULEAHDf 
echo "Downloaded MASIVE. Untarring and deleting archive..."
mv masive.tar.gz ./data/masive.tar.gz
tar -xzf ./data/masive.tar.gz
rm -f ./data/masive.tar.gz

# Download MASIVE models
gdown 1fQWvZFtUWGN9MTNmdQWEd0YPjMxKtDRL
gdown 1LZ-fK5R3kpfW8Q0dzanOX-7l7gB1KuvG
echo "Downloaded mt5 models. Untarring and deleting archives..."
mv en_random_mt5_large.tar.gz ./models/en_random_mt5_large.tar.gz
mv es_random_mt5_large.tar.gz ./models/es_random_mt5_large.tar.gz 
tar -xzf ./models/en_random_mt5_large.tar.gz
tar -xzf ./models/es_random_mt5_large.tar.gz
rm -f ./models/en_random_mt5_large.tar.gz
rm -f ./models/es_random_mt5_large.tar.gz
