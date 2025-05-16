import re

# Read the file
with open("randomized_conds.txt", "r") as f:
    lines = f.readlines()

# Parse each line to extract test name and time
parsed_data = []
for line in lines:
    match = re.match(r"Time taken for (.*?): ([\d.]+) seconds", line.strip())
    if match:
        test_name = match.group(1)
        time_taken = float(match.group(2))
        parsed_data.append((test_name, time_taken))

# Sort by time
sorted_data = sorted(parsed_data, key=lambda x: x[1])

# Output the sorted results
for test_name, time_taken in sorted_data:
    print(f"{test_name}: {time_taken:.4f} seconds")