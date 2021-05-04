import csv
import os
import re
import time
from functools import reduce
from google_trans_new import google_translator as Translator
import bs4
import pandas as pd
import requests

from Core import bs4utils, datautils

__per_site_api_path = os.path.dirname(__file__)
__root_path = os.path.dirname(__per_site_api_path)
__core_path = os.path.join(__root_path, "Core")
__data_path = os.path.join(__root_path, "Data")
__database_file = os.path.join(__data_path, "database.csv")
__zipcode_file = os.path.join(__data_path, "zipcodes.csv")
__logic_immo_url = os.path.join(__data_path, "logic_immo_url.csv")
__logic_immo_raw_data = os.path.join(__data_path, "logic_immo_raw_data.csv")
__logic_immo_data = os.path.join(__data_path, "logic_immo_clean_data.csv")
__root_url = r"https://www.logic-immo.be"
__addresses = set()
__pattern = re.compile(r'/fr/vente/.+/[a-zA-Z]+-[0-9]+/.+.html')
__translator = Translator()
__pattern = re.compile(r'/fr/vente/.+/.+/.+\.html')

__zipcode_df = pd.read_csv(__zipcode_file)
__headers = {
    # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    # "Accept-Encoding": "gzip, deflate",
    # "Accept-Language": "en-GB,en-US;q=0.9,en;q=0.8",
    # "Dnt": "1",
    # "Host": "httpbin.org",
    # "Upgrade-Insecure-Requests": "1",
    # "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9",
    # "Accept-Encoding": "gzip, deflate",
    # "Accept-Language": "fr-FR,en-US;q=0.9,en;q=0.8",
    # "Dnt": "1",
    # "Host": "httpbin.org",
    # "Upgrade-Insecure-Requests": "1",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36",
    # "X-Amzn-Trace-Id": "Root=1-5ee7bae0-82260c065baf5ad7f0b3a3e3"
}


def search_for_urls():
    old_len_address = 0
    for i, row in __zipcode_df.iterrows():
        print(i)
        print(row["local"])
        page = 1
        found = True
        text = "all"
        while found:
            found = False
            url = __root_url + f'/fr/vente/immo-a-vendre/{text}-{row["zipcode"]},{page},--------16776966-,---,---.html'
            response = requests.get(url, headers=__headers)
            if response.status_code == 500 or response.status_code == 503:
                while response.status_code == 500 or response.status_code == 503:
                    response = requests.get(url)
                    time.sleep(1)
            if response.status_code in (503, 500):
                while response.status_code in (500, 503):
                    response = requests.get(url)
                    time.sleep(1)

            buy_soup = bs4.BeautifulSoup(response.content, features="html.parser")
            for link in (f.get("href") for f in buy_soup.findAll("a")):
                if link:
                    if re.search(__pattern, link):
                        found = True
                        __addresses.add(__root_url + link)

            if not found:
                text = row["local"].lower().replace(" ", "-").replace("é", "e").replace("è", "e").replace("â", "a")
                text = text.replace("ç", "c").replace("û", "u").replace("ê", "e").replace("î", "i").replace("'", "")
                text = text.replace("ô", "o")
                url = __root_url + f'/fr/vente/immo-a-vendre/{text}-{row["zipcode"]},{page},--------16776966-,---,---.html'

                response = requests.get(url, headers=__headers)
                if response.status_code in (503, 500):
                    while response.status_code in (500, 503):
                        response = requests.get(url)
                        time.sleep(1)
                buy_soup = bs4.BeautifulSoup(response.content, features="html.parser")

                for link in (f.get("href") for f in buy_soup.findAll("a")):
                    if link:
                        if re.search(__pattern, link):
                            found = True
                            __addresses.add(__root_url + link)

            print(url, response.status_code, page, len(__addresses))
            # time.sleep(random.random()/4)
            page += 1

            if len(__addresses) > old_len_address + 1000:
                old_len_address = len(__addresses)
                with open(__logic_immo_url, 'wt+', newline='', encoding="utf-8") as url_file:
                    f_w = csv.writer(url_file)
                    f_w.writerows(([address] for address in sorted(__addresses)))


def search_raw_infos():
    with open(__logic_immo_url, "rt", encoding='utf-8') as url_file:
        with open(__logic_immo_raw_data, "w+", encoding="utf-8", newline="") as text_file:
            f_r = csv.reader(url_file)
            f_w = csv.writer(text_file)
            for url in f_r:
                url = url[0]
                response = requests.get(url, headers=__headers)
                if response.status_code in (503, 500):
                    while response.status_code in (500, 503):
                        response = requests.get(url)
                        time.sleep(1)
                print((url, response.status_code))

                soup = bs4.BeautifulSoup(response.content, features="html.parser")
                title = ''
                desc = ''
                icons = ""
                for element in soup.findAll("h1", class_="c-details_title c-details_title--primary"):
                    title += bs4utils.extract_text_from_tag(element).replace("\n", " ").replace('"', "").replace("'",
                                                                                                                 '')

                for element in soup.findAll('p', class_="js-description"):
                    raw_desc = bs4utils.extract_text_from_tag(element).replace("\n", " ").replace('"', "").replace("'", '')
                    while True:
                        try:
                            if __translator.detect(str(raw_desc))[0] != "fr":
                                raw_desc = str(__translator.translate(str(raw_desc), lang_tgt="fr"))
                            desc += raw_desc
                            break
                        except:
                            print("Translation failed, retrying")
                            time.sleep(0.2)
                for element in soup.findAll("ul", id="property-details-icons"):
                    for link in element.findAll("li", attrs={"data-toggle": 'tooltip'}):
                        icons += link.get("title") + bs4utils.extract_text_from_tag(link).replace("\n", " ") + ','

                title = title.replace("\n", " ").replace("\r", " ").replace(",", ".")
                desc = desc.replace("\n", " ").replace("\r", " ").replace(",", ".")
                icons = icons.replace("\n", " ").replace("\r", " ").replace('"', "").replace("'", '').split(",")

                f_w.writerow([url, title, desc] + icons)


def create_data_entry(datarow):
    new_entry = datautils.DataStruct.DATAFRAMETEMPLATE
    datarow[0] = datarow[0].lower()
    new_entry["Url"] = [datarow[0]]
    new_entry["Source"] = ["logic-immo.be"]
    new_entry["Zip"] = [int(datarow[0].split('/')[6].split("-")[-1])]
    localities = datautils.DataStruct.get_locality(new_entry["Zip"].values[0])
    local = ""

    if len(localities) == 1:
        local = localities[0].lower()
    else:
        for locality in localities:
            if datarow[1].lower().find(locality.lower()) != -1:
                local = locality.lower()
                break

    new_entry["Locality"] = local
    new_entry["Type of sale"] = ["vente"]
    if datarow[0].find("appartements") != -1:
        new_entry["Type of property"] = ["appartment"]
    elif datarow[0].find("maisons") != -1:
        new_entry["Type of property"] = ["maison"]
    else:
        new_entry["Type of property"] = [""]

    new_entry["Subtype of property"] = [datarow[1].split(" à vendre à")[0].split(" ")[-1].lower()]
    if not new_entry["Subtype of property"].values[0]:
        text = datarow[0]
        subtype = re.search('[0-9]{4}/([a-zA-Z]+)-[0-9]*', text)
        if subtype:
            new_entry["Subtype of property"] = subtype.groups()[0]

    datarow[1] = " ".join(bs4.UnicodeDammit(datarow[1]).unicode_markup.split())
    match_price = re.search("([0-9]{0,3}) ?([0-9]{0,3}) ?([0-9]{1,3}) ?0?€", datarow[1])
    price = 0
    if match_price:
        for i, group in enumerate(match_price.groups()[::-1]):
            if group != "":
                price += (1000**i) * int(group)
    new_entry["Price"] = [price]
    new_entry["Fully equipped kitchen"] = [False]
    for element in datarow:
        kitchen_all_match = re.search(r"(pas|sans|avec)? ?(de|une)? ?cuisine ([a-zA-Z\- ]*)* ?équipé", element.lower())
        if kitchen_all_match:
            for group in kitchen_all_match.groups():
                if group:
                    kitchen_bad_match = re.search("(sans|non|semi-é|semi é|partiellement)", group)
                    if kitchen_bad_match:
                        break
                    else:
                        new_entry["Fully equipped kitchen"] = [True]

    new_entry["Furnished"] = [False]
    for element in datarow:
        furnished_all_match = re.search("(avec|sans|partiellement|sur)? ?[^m]+(meublé|meuble)}", element.lower())
        if furnished_all_match:
            print(furnished_all_match)


    new_entry["Area"] = [0]
    pr = list()
    try:
        new_entry["Area"] = [int(datarow[1].split(".")[1].split(" m²")[0])]
    except IndexError:
        for element in datarow:
            m = re.search("surface ([0-9]+) m²", element.lower())
            pr.append((element, "Surface ([0-9]+) m²"))
            if m:
                new_entry["Area"] = [int(m.groups()[0])]
                break
        if new_entry["Area"].values[0] == 0:
            for element in datarow:
                pr.append((element, "([0-9]+) m²[a-zA-Z ]*hab"))
                m = re.search("([0-9]+) m²[a-zA-Z ]*hab", element.lower())
                if m:
                    area = 0
                    for group in m.groups():
                        area = int(group) if int(group) > area else area
                    new_entry["Area"] = [area]
                    break

        if new_entry["Area"].values[0] == 0:
            for element in datarow:
                pr.append((element, "surface du living ([0-9]+) m²"))
                m = re.search("surface du living ([0-9]+) m²", element.lower())
                if m:
                    new_entry["Area"] = [int(m.groups()[0])]
                    break

        if new_entry["Area"].values[0] == 0:
            for element in datarow:
                pr.append((element, "superficie totale [a-zA-Z ]*([0-9]+) m²"))
                m = re.search("superficie totale [a-zA-Z ]*([0-9]+) m²", element.lower())
                if m:
                    new_entry["Area"] = [int(m.groups()[0])]
                    break
        # if new_entry["Area"].values[0] == 0:
        #     print(pr)
    return new_entry


def create_data_csv():
    with open(__logic_immo_raw_data, "rt", encoding="utf-8") as data_file:
        with open(__logic_immo_data, "w+", encoding="utf-8") as data_final_file:
            f_r = csv.reader(data_file)
            data = datautils.DataStruct.DATAFRAMETEMPLATE
            for line in f_r:
                pd.options.display.max_columns = 30
                data = data.append(create_data_entry(line), ignore_index=True)

            data.dropna(how="any", subset=["Subtype of property"], inplace=True)
            data.to_csv(data_final_file)


if __name__ == "__main__":
    # search_for_urls()
    search_raw_infos()
    create_data_csv()
