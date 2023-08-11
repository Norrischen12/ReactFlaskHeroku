import sys
import os
import pandas as pd


def main():
    # Check if the correct number of arguments is provided
    if len(sys.argv) != 3:
        print("Usage: python your_script.py <arg1> <arg2>")
        return

    # Get the arguments
    arg1 = sys.argv[1]
    arg2_path = sys.argv[2]

    # Check if the provided file exists and is an XLSX file
    if not os.path.isfile(arg2_path) or not arg2_path.lower().endswith('.xlsx'):
        print("Error: Argument 2 must be a valid XLSX file.")
        return

    # Your script logic here
    # Use arg1 and arg2_path as needed in your script

    # Example: Print the arguments and read the Excel file
    # print("Argument 1:", arg1)
    # print("Argument 2 (Excel file path):", arg2_path)

    # Read the Excel file using pandas
    df = pd.read_excel(arg2_path)

    html_table = df.to_html(index=False)

    print(html_table)


if __name__ == "__main__":
    main()
