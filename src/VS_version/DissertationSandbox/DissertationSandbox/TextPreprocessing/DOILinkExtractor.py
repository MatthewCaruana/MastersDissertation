import requests
import time
from bs4 import BeautifulSoup as bs
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains


DRIVER_PATH = 'E:\Development\MastersDissertation\src\VS_version\DissertationSandbox\DissertationSandbox\ChromeDriver\chromedriver.exe'
driver = webdriver.Chrome(executable_path=DRIVER_PATH)

default_page = "https://www.gov.mt/en/Government/DOI/Press%20Releases/Pages/default.aspx"
driver.get(default_page)

links = []

for page in range(2, 25000):
    link_buttons = driver.find_elements(By.CLASS_NAME, "cbs-pictureImgLink")
    for link_button in link_buttons:
        links.append(link_button.get_attribute("href"))

    print("Page " + str(page) + " has been completed!")


    pagination_element = driver.find_element(By.CLASS_NAME, "pagination-next")
    next_page_button = pagination_element.find_element(By.CLASS_NAME, "ms-promlink-button")
    ActionChains(driver).move_to_element(pagination_element).click(next_page_button).perform()
    time.sleep(2)


file = open("DOILinks.txt",'w')
for link in links:
    file.write(link + "\n")

file.close()