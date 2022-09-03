import json
import requests
from bs4 import BeautifulSoup as bs

headers = { 'Host' : "www.gov.mt",
            'Accept-Encoding': "gzip, deflate, br",
            'Connection' : "keep-alive",
            'User-Agent':"requests/2.26.0"}

links_file = open("DOILinks.txt", 'r')

press_releases = []
i= 1

for link in links_file:
    req = requests.get(link[:-1],headers = headers)
    soup = bs(req.text, 'html.parser')

    press_release = soup.find('div',attrs = {'class','press-con'})

    press_release_number = soup.find('h6', attrs={'class', 'release-no'})
    press_release_date = soup.find('p', attrs={'class', 'date'})
    press_release_title = soup.find("h2", attrs={'class', 'theme-color-2'})

    press_release_content = soup.find('div', attrs={'id': 'ctl00_PlaceHolderMain_ctl06__ControlWrapper_RichHtmlField'})

    single_press_release = {"Number": press_release_number.text, "Date": press_release_date.text, "Title": press_release_title.text, "Content": press_release_content.text}

    press_releases.append(single_press_release)
    print("Finished number " + str(i))
    i= i + 1


file = open("DOIContent.json", 'w', encoding="utf-8")
press_release_json = json.dump(press_releases, file, ensure_ascii=False)

file.close()