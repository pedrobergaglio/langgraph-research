import requests

def fetch_page(url):
    response = requests.get(url)
    return response.text

# Example usage
if __name__ == "__main__":
    url = 'https://www.google.com'
    page_content = fetch_page(url)
    print(page_content)