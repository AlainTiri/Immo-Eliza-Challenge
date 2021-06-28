import csv
import datetime
import os
from random import random

from bs4 import BeautifulSoup

from Core.datautils import DataStruct
import re
import pandas as pd

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager

from time import sleep
from typing import List, Dict
import requests
import json

from PerSiteApi.headers import headers_list


class ScrapedSite:

    def __init__(self, site_name: str, site_url: str):
        self.data_immo: Dict = {}
        """self.url_error: List[str] = []
        self.zip_code: List = DataStruct().get_zipcode_data()
        """
        self._name_ = site_name
        self._site_url_ = site_url
        self._per_site_api_path = os.path.dirname(__file__)
        self._root_path = os.path.dirname(self._per_site_api_path)
        self._data_path = os.path.join(self._root_path, "Data")
        self._data_my_path = os.path.join(self._data_path, self._name_)
        self.path_results_search = os.path.join(self._data_my_path, "url_results_search.csv")
        self.path_items = os.path.join(self._data_my_path, "url_items.csv")
        self.path_scraped_data = os.path.join(self._data_my_path, "scraped.csv")
        if not (os.path.exists(self._data_path)):
            os.mkdir(self._data_path)
        if not (os.path.exists(self._data_my_path)):
            os.mkdir(self._data_my_path)
        """
        s = pd.read_csv(self.path_results_search)["0"] #TODO : what if failed
        self.url_results_search = set(s)
        s = pd.read_csv(self.path_items)["0"]
        self.url_immo = set(s)
        self.datas_immoweb = pd.read_csv(self.path_scraped_data, index_col=0)
        self.r_num_page = re.compile("\d{1,3}$")
        self.r_zip_code = re.compile("/\d{4}/")
        self.url_vente = "https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre?countries=BE&orderBy=cheapest&postalCodes=BE-"
        self.immo_path: Dict = {
            "Zip": ["customers", "location","postalCode"],
            "Locality": ["customers", "location", 'locality'],
            "Type of property": ["property", "type"],
            "Subtype of property": ["property", "subtype"],
            "Price": ["transaction", "sale", "price"],
            "Type of sale": ["transaction", "subtype"],
            "Number of rooms": ["property", "bedroomCount"],
            "Area": ["property", "netHabitableSurface"],
            "Fully equipped kitchen": ["property", "kitchen","type"],  # !!! Kitchen peut être null !!
            "Furnished": ["transaction", "sale", "isFurnished"],
            "Open fire": ["property", "fireplaceExists"],
            "Terrace": ["property", "hasTerrace"],
            "Terrace Area": ["property", "terraceSurface"],
            "Garden": ["property", "hasGarden"],
            "Garden Area": ["property", "gardenSurface"],
            "Surface of the land": ["property", "land", "surface"],  # !!! land peut être null !
            "Surface area of the plot of land": ["property", "land","surface"],  # !!! land peut être null !
            "Number of facades": ["property", "building", "facadeCount"],
            "Swimming pool": ["property", "hasSwimmingPool"],
            "State of the building": ["property", "building", "condition"]
        }
        """
        user_agent_desktop = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                             'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 ' \
                             'blob/537.36'

        self.headers = random.choice(headers_list)
        # self.driver = webdriver.Firefox()

    # retourne le milieu du texte entre les deux balises et le texte qui suit
    def coupepage(self, texte, debut, fin, n=1):
        debutT = texte.index(debut) + len(debut)
        texte = texte[debutT:]
        finT = texte.index(fin)

        if finT:
            return texte[:finT], texte[finT:]
        else:
            return False, False

    def _finditem(self, obj, key):
        if key in obj: return obj[key]
        for k, v in obj.items():
            if isinstance(v, dict):
                return self._finditem(v, key)  # added return statement

    def json_to_dic(self, json_immo: dict):
        # print("immo_path", self.immo_path)
        new_sale = {}
        for key, path in self.immo_path.items():
            n = len(path)
            i = 0
            element = json_immo[path[i]]
            if type(element) is list:
                element = element[0]

            i += 1
            while i < n and element is not None:
                element = element[path[i]]
                i += 1
            new_sale[key] = element
        return new_sale

    def scan_page_bien_immobilier(self, url: str) -> Dict:
        print(url)
        page = requests.get(url)
        texte = page.text
        texte, suite = self.coupepage(texte, "window.classified = ", "};")
        texte += "}"
        json_immo = json.loads(texte)

        new_sale = self.json_to_dic(json_immo)
        if new_sale["Zip"] is None:
            zip = re.search(self.r_zip_code, url).group(0)
            zip = zip[1:-1]
            new_sale["Zip"] = int(zip)

        if new_sale["Locality"] is None:
            new_sale["Locality"] = DataStruct.get_locality(new_sale["Zip"])[0]

        new_sale["Url"] = url
        new_sale["Source"] = "Immoweb"
        print("type et new_sale", type(new_sale), new_sale)

        return pd.DataFrame(new_sale, index=[len(self.datas_immoweb.index)])

    def loop_immo(self):

        i = 0
        max = 75000
        for url in self.url_immo:
            if url not in self.datas_immoweb["Url"]:
                new_sale = self.scan_page_bien_immobilier(url)
                self.datas_immoweb = self.datas_immoweb.append(new_sale)
                i += 1
                sleep(0.3)
            if i > max:
                break

        print(type(self.datas_immoweb), self.datas_immoweb)
        self.save_data_to_csv()

    def save_data_to_csv(self):
        self.datas_immoweb.to_csv(self.path_scraped_data)

    def run(self):
        self._generator_db_url()
        # self._scan_page_list('https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre?countries=BE&postalCodes=BE-1348')
        # self._add_set_to_csv()
        # self.scan_page_bien_immobilier('https://www.immoweb.be/fr/annonce/kot/a-vendre/bruxelles-ville/1000/9303884?searchId=60914b580d2b7')
        self._save_set_to_csv()
        self.driver.close()
        self.loop_immo()
        self.save_data_to_csv()
        print("fini")

    def _save_set_to_csv(self):
        with open(self.path_results_search, 'w') as file:
            df = pd.DataFrame(self.url_results_search)
            df.to_csv(file)

        with open(self.path_items, "w") as file:
            df = pd.DataFrame(self.url_immo)
            df.to_csv(file)

    # https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre?countries=BE&orderBy=cheapest&postalCodes=BE-1341,1348
    def discover(self, url_postfix_category: str, url_postfix_fields: str, search_pager: str, pages=1):
        """
        Sets the url for searching 
        :param pages:
        :param url_postfix_category:  'fr/recherche/maison-et-appartement/a-vendre'
        :param url_postfix_fields: '?countries=BE&orderBy=cheapest&postalCodes=BE-1341,1348'
        :param search_pager: '&page='
        :return: list of url to scan for items
        """
        # todo : build dict for category and fields
        self.__url_postfix_category__ = url_postfix_category
        self.__url_postfix_fields__ = url_postfix_fields
        self.__search_pager__ = search_pager
        final_url = f"{self._site_url_}{url_postfix_category}{url_postfix_fields}{search_pager}"
        search_url_list = [f"{final_url}{i}" for i in range(pages+1)]
        return search_url_list

    def _scan_page_list(self, url: str, num_pages: int = 1, pages=1) -> bool:
        """
        Scan the page with result of research.
        :param url: An String url without page=xx
        :param num_pages: Number of the page to scan
        :return: Now, I dont know.
        """

        pager = f"&page={num_pages}"
        url_paged = url + pager
        pagination = [0]
        if url_paged not in self.url_results_search:
            self.url_results_search.add(url_paged)
            self.driver.get(url_paged)
            text = self.driver.page_source

            if "Désolé. Aucun résultat trouvé." in text:
                print(url_paged, " pas trouvé")
                return False
            elif "pagination__item" in text:
                li_pagination_item = self.driver.find_elements_by_css_selector('li.pagination__item')
                pagination = [1]
                for element in li_pagination_item:
                    number = re.search(self.r_num_page, element.text)
                    if number is not None:
                        pagination.append(int(number.group(0)))
                print(pagination)
                if max(pagination) == 333:
                    print("333 pages a scaner => url passée : ", url_paged)
                    return False

            a_elements = self.driver.find_elements_by_css_selector('a.card__title-link')
            for element in a_elements:
                self.url_immo.add(element.get_attribute('href'))
            print(len(self.url_immo), " adresses url de biens")

            #  Section who verify if we must turn to next page
            if max(pagination) == num_pages or pagination[0] == 0:
                return True
            else:
                self._scan_page_list(url, num_pages + 1)

    def find_items(self,item_link):
        link_list=[]
        r = requests.get(item_link,
                         headers=self.headers)  # http://validate.perfdrive.com/immovlan/captcha?ssa=caee399c-1aad-40c8-b615-a7fc6af6c779&ssb=mh0pep5i6pihbb24dm3h0k11k&ssc=http%3a%2f%2fimmo.vlan.be%2ffr%2fimmobilier%3fpropertytypes%3dappartement%2cmaison%2cbusiness%2cterrain%2cgarage%2cbien-d%2527investissement%2ckot%2cdivers%26transactiontypes%3da-vendre%2ca-louer%2cen-vente-publique%2cen-colocation%26propertysubtypes%3dappartement%2crez-de-chaussee%2cduplex%2cpenthouse%2cstudio%2cloft%2ctriplex%2cmaison%2cvilla%2cimmeuble-mixte%2cmaison-de-maitre%2cfermette%2cbungalow%2cchalet%2cchateau%2csurface-commerciale%2cbureaux%2csurface-industrielle%2cfonds-de-commerce%2cferme%2cterrain-a-batir%2cterrain%2cterrain-agricole%2cparking%2cgarage%2cmaison-de-rapport%2ckot-etudiant%2cbien-divers%26countries%3dbelgique%26noindex%3d1%26pageOffset%3d1&ssd=970959273986341&sse=olapjpemfpm@ggc&ssf=eb0f8f4c4fcb66138e2a0b11b2df4950e89d1b7e&ssg=2faf2114-77e1-4b80-b1e9-4d0c95be0d68&ssh=f205d560-d340-40d2-9250-da0d2d778adb&ssi=971a1329-be01-ef47-d925-95a95e54fd43&ssj=4e22da2e-2c6e-424e-a5dc-de03652e22d6&ssk=Immovlan-support@shieldsquare.com&ssl=970959273986341&ssm=97095927398634125109709592739863&ssn=eb0f8f4ca5d1f5d57d8406ea13b0eb0f8f4ceb0f8&sso=eb0f8eb0f8f4c00a330b2f5dce4a1fe5eeb0f8f4c&ssp=970959273997095162029709592739&ssq=970959210942970959273970959273986341&ssr=OTQuMTA0LjE5Ny4yMTM=&sss=Mozilla/5.0%20(compatible;%20Yahoo!%20Slurp;%20http://help.yahoo.com/help/us/ysearch/slurp)&sst=Mozilla/5.0%20(Windows%20NT%2010.0;%20Win64;%20x64)%20AppleWebKit/537.36%20(KHTML,%20like%20Gecko)%20Chrome/80.0.3987.149%20Safiri/537.36&ssu=Mozilla/5.0%20(compatible;%20Yahoo!%20Slurp;%20http://help.yahoo.com/help/us/ysearch/slurp)&ssv=o3pp9pv@2pl342s&ssw=&ssx=970959273986341&ssy=olapjpemfpm@ggcdiod@albblcnpekfaojknblho&ssz=eb0f8f4c4fcb661
        if "http://validate.perfdrive.com/immovlan/captcha" in r.url:
            print("captchah")
            print(r.url)
            r = requests.get(input("url ?"),headers = self.headers)
        soup = BeautifulSoup(r.content, 'lxml')
        for soup_links in soup.findAll('a'):
            link = soup_links.get("href")
            if 'a-vendre' in link and '/fr/detail/' in link:
                link_list.append(link)
        return link_list

    def site_index(self):
        vlan_pf_cat = 'fr/immobilier'
        vlan_pf_fld = '?propertytypes=appartement,maison,business,terrain,garage,' \
                      'bien-d%27investissement,kot,divers&transactiontypes=a-vendre,a-louer,en-vente-publique,' \
                      'en-colocation&propertysubtypes=appartement,rez-de-chaussee,duplex,penthouse,studio,loft,triplex,maison,' \
                      'villa,immeuble-mixte,maison-de-maitre,fermette,bungalow,chalet,chateau,surface-commerciale,' \
                      'bureaux,surface-industrielle,fonds-de-commerce,ferme,terrain-a-batir,terrain,terrain-agricole,' \
                      'parking,garage,maison-de-rapport,kot-etudiant,bien-divers&countries=belgique'
        vlan_pf_pager = '&noindex='
        to_scrape_df = pd.DataFrame(vlan.discover(vlan_pf_cat, vlan_pf_fld, vlan_pf_pager,168))
        to_scrape_df["done"] = 0
        if os.path.exists(self.path_results_search):
            to_scrape_df_file = pd.read_csv(self.path_results_search)
            to_scrape_df = pd.concat([to_scrape_df, to_scrape_df_file])
        # load from file and merge
        items_df = pd.DataFrame()
        for to_scrape in to_scrape_df[to_scrape_df[
                                          "done"] == 0].iloc[:,0]:  # scrape only for done = 0 , what for old data save when finished or after interruption
            new_data_df = pd.DataFrame(self.find_items(to_scrape))
            if items_df is None:
                items_df = new_data_df
            else:
                items_df = pd.concat([items_df, new_data_df])
            # ittems_df["done"]  = 0
            # for item in in items_df:
            #     scrape(item)
            to_scrape_df["done"] = datetime.datetime.now()
            items_df["done"] = 0
            items_df["from"] = to_scrape

        to_scrape_df.to_csv(self.path_results_search)  # file must be overwritten
        if os.path.exists(self.path_items):
            items_df_file = pd.read_csv(self.path_items)
            items_df = pd.concat([items_df, items_df_file])
        items_df.to_csv(self.path_items)


def scrape(item):
    item_url = __root_url + links
    time.sleep(random() + 0.5)
    r = requests.get(item_url, headers=headers)
    soup = BeautifulSoup(r.content, 'lxml')
    infos = soup.findAll("div", {"id": re.compile("^collapse.*$")})
    for data_from_collapsed_section in infos:
        """
        #"collapse_general_info"
        <div class="col-6">Etat du bien</div><div class="col-6 text-right">À rénover</div><div class="small-border w-100"></div>
        data = {'First_Name': ['Jeff','Tina','Ben','Maria','Rob'],
            'Last_Name':['Miller','Smith','Lee','Green','Carter'],
            'Age':[33,42,29,28,57]
                }
            df = pd.DataFrame(data, columns = ['First_Name','Last_Name','Age'])
        """
        column_name = data_from_collapsed_section.find("div", {"class": re.compile("^col-[0-9]+$")})
        value = data_from_collapsed_section.find("div", {"class": re.compile("^col-[0-9]+ text.*$")})
        if column_name is not None:
            if column_name not in list_column:
                list_column.append(column_name.text.strip())


if __name__ == "__main__":
    # immoweb = ScrapedSite("Immoweb","https://www.immoweb.be/")
    # immoweb.discover('fr/recherche/maison-et-appartement/a-vendre','?countries=BE&orderBy=cheapest&postalCodes=BE-1341,1348','&page=')
    # immoweb.run()
    vlan = ScrapedSite("Vlan", "https://immo.vlan.be/")
    vlan.site_index()  # after this , path_results_search &path_items  should be populated
    # vlan.scrape_items()
