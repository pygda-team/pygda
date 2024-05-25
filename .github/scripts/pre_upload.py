import os

def main():
    api_token = os.getenv('PYPI_API_TOKEN')
    if not api_token:
        raise ValueError("PyPI API token not found in environment variables")

    # Use the API token for your operations
    print("API token acquired successfully")
    # Your code that uses the API token goes here
    # For example, preparing the package for upload

if __name__ == "__main__":
    main()