from collections import Counter

# Function to calculate the number of common letters between two names
def common_letters(name1, name2):
    # Create Counters for each name
    counter1 = Counter(name1)
    counter2 = Counter(name2)
    
    # Combine the Counters to find common letters
    common_counter = counter1 & counter2
    
    # Return the total count of common letters
    return common_counter.total()

# Names to compare
name1 = 'Johnny'
name2 = 'Bonny'

# Calculate the number of common letters
common_count = common_letters(name1, name2)

# Print the result
print(f"The number of common letters between '{name1}' and '{name2}' is: {common_count}")

