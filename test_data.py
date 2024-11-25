import polars as pl

# Create sample data
data = {
    "intent": ["greeting", "farewell", "help_request"],
    "text": ["hello there", "goodbye", "can you help me"]
}

# Create and save DataFrame
df = pl.DataFrame(data)
df.write_csv("sample_data.csv") 