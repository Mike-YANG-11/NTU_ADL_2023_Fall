import re
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


options = Options()
options.add_experimental_option("detach", True)

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options,
)

book_title = "Thinking, Fast and Slow"
# book_title = "The Lost Bookshop:"
search_title = "Amazon+" + book_title.replace(" ", "+")
driver.get(f"https://www.google.com.tw/search?q={search_title}")

# Click the first Amazon link on Google search result
links = driver.find_elements("xpath", "//a[@href]")
for link in links:
    # check if the link is an Amazon link and the first word in the book title is in the link
    if (
        # re.split(" |,|:|-|?", book_title)[0].lower() in link.get_attribute("href").lower()
        "amazon.com" in link.get_attribute("href")
        and "/dp/" in link.get_attribute("href")
    ):
        print(link.get_attribute("href"))
        responese = requests.get(link.get_attribute("href"))
        link.click()
        break

# Expand the book introduction
driver.find_element("xpath", "//div[@id='bookDescription_feature_div']//span[@class='a-expander-prompt']").click()
# Get the book introduction
book_intro = driver.find_element(
    "xpath", "//div[@id='bookDescription_feature_div']//div[@data-a-expander-name='book_description_expander']//div"
)

# Process the book introduction
book_intro = book_intro.text.replace("\n", ". ").replace("  ", " ")
# Delete the special characters except for the period and dot
book_intro = re.sub("[^a-zA-Z0-9 . ,]", "", book_intro)
# book_intro = re.sub("[^a-zA-Z0-9 ]", "", book_intro)
print(book_intro)
