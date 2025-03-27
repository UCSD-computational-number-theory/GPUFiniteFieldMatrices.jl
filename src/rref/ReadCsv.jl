using CSV

# Read CSV file into a DataFrame
df = CSV.read("your_file.csv")

# Convert DataFrame to a matrix
matrix_data = convert(Matrix, df)