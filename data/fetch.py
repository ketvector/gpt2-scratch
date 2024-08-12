from bs4 import BeautifulSoup
import requests

def fetch_html_page(url):
    try:
        # Send a GET request to the specified URL
        response = requests.get(url)
        
        # Raise an exception for bad status codes
        response.raise_for_status()
        
        # Return the HTML content if the request was successful
        return response.text
    except requests.RequestException as e:
        print(f"An error occurred: {e}")
        return None


base_url = "https://paulgraham.com/"
html_doc = fetch_html_page(f"{base_url}/articles.html")

soup = BeautifulSoup(html_doc, "lxml")

all_links = soup.find_all('a')
all_links = all_links[0:10]

def get_contents(sp):
    try:
        section = []
        font = str(sp.findAll('table', {'width':'435'})[0].findAll('font')[0])
        if not 'Get funded by' in font and not 'Watch how this essay was' in font and not 'Like to build things?' in font and not len(font)<100:
            content = font
        else:
            content = ''
            for par in sp.findAll('table', {'width':'435'})[0].findAll('p'):
                content += str(par)

        for p in content.split("<br /><br />"):
            section.append(p)

        #exception for Subject: Airbnb
        for pre in sp.findAll('pre'):
            section.append(pre)
        
        return section
    except:
        print("EXCEPTION")
        pass

ans = []
for link in all_links:
    l = link.get('href')
    if l.endswith(".html"):
        url = f"{base_url}{link.get('href')}"
        data = fetch_html_page(url)
        sp = BeautifulSoup(data)
        y = BeautifulSoup(data, "lxml").text
        ans.append(y)
        

        
        


