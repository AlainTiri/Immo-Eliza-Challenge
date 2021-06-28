import os

from Core.datautils import DataStruct
import re
import pandas as pd

import requests
from selenium import webdriver
from typing import List, Dict

import queue
import concurrent.futures

import json
import time


class ThreadPoolExecutorWithQueueSizeLimit(concurrent.futures.ThreadPoolExecutor):
    def __init__(self, maxsize=5, *args, **kwargs):
        super(ThreadPoolExecutorWithQueueSizeLimit, self).__init__(*args, **kwargs)
        self._work_queue = queue.Queue(maxsize=maxsize)


def coupe_page(text, start, end):
    start_text: int = text.index(start) + len(start)
    text: str = text[start_text:]
    end_text: int = text.index(end)

    if end_text:
        return text[:end_text], text[end_text:]
    else:
        return False, False


class Immoweb:

    def __init__(self):
        self.start = time.perf_counter()
        self.start_loop = time.perf_counter()

        self.data_immo: Dict = {}
        self.zip_code: List = DataStruct().get_zipcode_data()

        self._per_site_api_path = os.path.dirname(__file__)
        self._root_path = os.path.dirname(self._per_site_api_path)
        self._data_path = os.path.join(self._root_path, "Data")
        self._data_immoweb_path = os.path.join(self._data_path, "data_immoweb")
        self.path_results_search = os.path.join(self._data_immoweb_path, "url_results_search.csv")
        self.path_immo = os.path.join(self._data_immoweb_path, "url_immo.csv")
        self.path_data_immoweb = os.path.join(self._data_immoweb_path, "datas_immoweb.csv")
        self.path_clusters = os.path.join(self._data_immoweb_path, "url_cluster.csv")
        self.path_errors = os.path.join(self._data_immoweb_path, "url_errors.csv")
        self.full_data = os.path.join(self._data_immoweb_path, "full_data.csv")

        s = pd.read_csv(self.path_results_search, index_col=0)["0"]
        self.url_results_search: set = set(s)
        s = pd.read_csv(self.path_immo, index_col=0)["0"]
        self.url_immo: set = set(s)
        s = pd.read_csv(self.path_clusters, index_col=0)["url"]
        self.url_from_clusters: set = set(s)
        self.url_of_clusters: set = set()
        s = pd.read_csv(self.path_errors, index_col=0)["url"]
        self.url_errors: set = set(s)
        self.datas_immoweb = pd.read_csv(self.path_data_immoweb, index_col=0)  # DataFrame: with all data collected
        self.url_of_datas = set(self.datas_immoweb.Url)

        self.url_immo.update(self.url_from_clusters)
        self.url_immo.discard(self.url_errors)
        self.url_immo.discard(self.url_of_datas)

        self.r_num_page = re.compile("\d{1,3}$")
        self.r_zip_code = re.compile("/\d{4}/")

        self.headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:20.0) Gecko/20100101 Firefox/20.0'}
        self.url_vente: str = "https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre?countries=BE&orderBy=cheapest&postalCodes=BE-"
        self.immo_path: Dict = {
            "Type of property": ["property", "type"],
            "Subtype of property": ["property", "subtype"],
            "Price": ["transaction", "sale", "price"],
            "Type of sale": ["transaction", "subtype"],
            "Number of rooms": ["property", "bedroomCount"],
            "Area": ["property", "netHabitableSurface"],
            "Fully equipped kitchen": ["property", "kitchen", "type"],  # !!! Kitchen peut être null !!
            "Furnished": ["transaction", "sale", "isFurnished"],
            "Open fire": ["property", "fireplaceExists"],
            "Terrace": ["property", "hasTerrace"],
            "Terrace Area": ["property", "terraceSurface"],
            "Garden": ["property", "hasGarden"],
            "Garden Area": ["property", "gardenSurface"],
            "Surface of the land": ["property", "land", "surface"],  # !!! land peut être null !
            "Surface area of the plot of land": ["property", "land", "surface"],  # !!! land peut être null !
            "Number of facades": ["property", "building", "facadeCount"],
            "Swimming pool": ["property", "hasSwimmingPool"],
            "State of the building": ["property", "building", "condition"],
            "last Modification Date": ["publication", "lastModificationDate"],
            "Region": ["property", "location", "region"],
            "Province": ["property", "location", "province"],
            "Arrondissement": ["property", "location", "district"],
            "Rue": ["property", "location", "street"],
            "Numero": ["property", "location", "number"],
            "Etage": ["property", "location", "floor"],
            "Latitude": ["property", "location", "latitude"],
            "Longitude": ["property", "location", "longitude"],
            "Is Agency": ["customers", "type"]
        }
        self.json_path_clusters: Dict = {
            "id": ["cluster", "units", "items", "id"],  # units = List[0], items = List[x]
            "subtype": ["cluster", "units", "items", "subtype"],  # units = List[0], items = List[x]
            "saleStatus": ["cluster", "units", "items", "saleStatus"]  # units = List[0], items = List[x]

        }
        self.data_layer_json: Dict = {
            "Zip": ["classified", "zip"]
        }

        ##### Lists used for the MultiThreading ######
        self.provinces_house_apartement: List = [
            ["apartment", "flandre-occidentale/province"],  # 209
            ["house", "hainaut/province"],  # 167
            ["house", "flandre-occidentale/province"],  # 161
            ["apartment", "bruxelles/province"],  # 148
            ["house", "flandre-orientale/province"],  # 147
            ["apartment", "anvers/province"],  # 146
            ["house", "anvers/province"],  # 132
            ["house", "liege/province"],  # 117 pages
            ["apartment", "flandre-orientale/province"],  # 110
            ["house", "brabant-flamand/province"],  # 94
            ["apartment", "brabant-flamand/province"],  # 72
            ["house", "limbourg/province"],  # 59
            ["apartment", "liege/province"],  # 55
            ["house", "bruxelles/province"],  # 52
            ["apartment", "hainaut/province"],  # 52
            ["house", "namur/province"],  # 49
            ["apartment", "limbourg/province"],  # 47
            ["house", "brabant-wallon/province"],  # 47
            ["house", "luxembourg/province"],  # 33
            ["apartment", "brabant-wallon/province"],  # 27
            ["apartment", "namur/province"],  # 18
            ["apartment", "luxembourg/province"]  # 12
        ]
        self.reloop_province: List = [
            ["house", "hainaut/province"],
            ["apartment", "bruxelles/province"]
        ]
        self.counter = 0
        self.counter_errors = 0

    # retourne le milieu du texte entre les deux balises et le texte qui suit

    def json_to_dic(self, json_immo: dict) -> Dict:
        new_sale: Dict = dict()
        for key, path in self.immo_path.items():
            n: int = len(path)
            i: int = 0
            element = json_immo[path[i]]
            i += 1

            while (i < n) and (element is not None):
                if type(element) is list:
                    element = element[0]
                element = element[path[i]]
                i += 1

            new_sale[key] = element

        return new_sale

    def scan_page_bien_immobilier(self, url: str) -> Dict:
        if url not in self.url_of_datas:
            page = requests.get(url)
            text: str = page.text
            new_property: Dict = {}
            zip_locality = re.split("/", url)
            new_property["Zip"] = zip_locality[8]
            new_property["Locality"] = zip_locality[7]
            new_property["Url"] = url
            new_property["Source"] = "Immoweb"

            if "404 not found" not in text:  # ThenAlain
                text, suite = coupe_page(text, "window.classified = ", "};")
                text += "}"
                json_immo = json.loads(text)
                # print("2", self.counter, url, new_property["Zip"], type(json_immo), json_immo)
                new_property.update(self.json_to_dic(json_immo))
                print(self.counter, "len", len(self.url_immo), url)
                self.counter += 1
                if self.counter % 1000 == 0:
                    print(self.counter, "nbr biens: ", len(self.datas_immoweb), "len", len(self.url_immo))
                self.datas_immoweb.append(new_property)
                return new_property
            else:  # If it's error 404
                self.url_errors.add(url)
                print("Error 404", url)
                return None

    def loop_immo(self):
        self.start_loop = time.perf_counter()
        self.clean_url_immo()
        self.url_of_datas = set(self.datas_immoweb.Url)
        self.url_immo.update(self.url_from_clusters)
        self.url_immo.discard(self.url_errors)
        self.url_immo.discard(self.url_of_datas)

        i = 0

        print("url Immo", len(self.url_immo))
        print("url of datas", len(self.url_of_datas))

        with ThreadPoolExecutorWithQueueSizeLimit(maxsize=20, max_workers=20) as executor:

            news_properties = executor.map(self.scan_page_bien_immobilier, self.url_immo)
            for property in news_properties:
                i += 1
                print(i)
                if property is not None:
                    print (i, "property", i)
                if i % 5000 == 0:
                    print(i, "save 5000")

        print(type(self.datas_immoweb), self.datas_immoweb)
        time.sleep(20)
        self.save_data_to_csv()
        print("saved")

        finish = time.perf_counter()
        print(f"fini loop_immo en {round(finish-self.start_loop, 2)} secondes")

        duration = 0.5  # seconds
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

        ask = input("Rerun ? (y/n)")
        rerun = ask != "n"
        print(rerun)
        if rerun:
            self.loop_immo()

    def run(self):
        print(self.datas_immoweb.shape)
        # self._generator_db_url()
        self.cluster()  # pour le testing, sinon, lancé dans generator url
        self.clean_url_immo()
        self._save_url_immo()
        print("delayer")
        time.sleep(1)

        self.loop_immo()
        print("delayer")
        time.sleep(20)
        #self.save_data_to_csv()
        finish = time.perf_counter()
        print(f"fini en {round(finish-self.start, 2)} secondes")

    def save_data_to_csv(self):
        duration = 0.5  # seconds
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))
        self.datas_immoweb.to_csv(self.path_data_immoweb)
        print("SAUVEGARDE A ", self.datas_immoweb.shape, "len url immo", len(self.url_immo), "len database", len(self.url_of_datas))
        print("len url from clusters", len(self.url_from_clusters))
        print(self.datas_immoweb.info())
        finish = time.perf_counter()
        print(f"Save at {round(finish-self.start_loop, 2)} secondes from start_loop")

    def _save_urls_results_to_csv(self):
        with open(self.path_results_search, 'w') as file:
            df = pd.DataFrame(self.url_results_search)
            df.to_csv(file)

    def _save_clusters(self):
        with open(self.path_clusters, 'w') as file:
            df = pd.DataFrame(self.url_from_clusters)
            df.to_csv(file)

    def _save_url_immo(self):
        # self.clean_url_immo()
        with open(self.path_immo, "w") as file:
            df = pd.DataFrame(self.url_immo)
            df.to_csv(file)

    def clean_url_immo(self):
        new_set = set()
        print("clean url_immo")
        for url in self.url_immo:
            new_url = url.split("?searchId=")[0]
            new_set.add(new_url)
        self.url_immo = new_set

    def find_url_in_clusters(self, url: str) -> List:
        page = requests.get(url)
        text: str = page.text
        print(type(text))
        if "404 not found" not in text:
            try:
                new_property: Dict = {}
                zip_locality = re.split("/", url)
                new_property["Zip"] = zip_locality[8]
                new_property["Locality"] = zip_locality[7]
                new_property["Url"] = url
                new_property["Source"] = "Immoweb"
                list_url = []

                print("cluster", url, text)
                text, suite = coupe_page(text, "window.classified = ", "};")
                text += "}"
                print(text, suite)
                json_immo: Dict = json.loads(text)
                print(json_immo)
                cluster = json_immo["cluster"]
                print(url, cluster)
                for unit in cluster["units"]:
                    for item in unit["items"]:
                        if item['saleStatus'] == "AVAILABLE":
                            new_url_property_cluster: str = f"https://www.immoweb.be/fr/annonce/{item['subtype']}/a-vendre/{new_property['Locality']}/{new_property['Zip']}/{item['id']}"
                            list_url.append(new_url_property_cluster)

                # print(len(list_url), type(list_url))
                return list_url
            except TypeError:
                print("Error de Type", url)
            except ValueError:
                print("Value Error", url)
                self.counter_errors += 1
                if (self.counter_errors % 100) == 0:
                    os.system('play -nq -t alsa synth {} sine {}'.format(0.2, 440))
                e = 500
                if (self.counter_errors % e) == 0:
                    print(f"{e} erreurs")
                    time.sleep(e)
                    os.system('play -nq -t alsa synth {} sine {}'.format(1, 440))

        else:
            print("Error 404", url)

# https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre?countries=BE&orderBy=cheapest&postalCodes=BE-1341
    def _scan_page_list(self, province: List):
        """
        Scan the page with result of research. Called in thread.
        :param province:
        :return: Closed when finished.
        """

        num_pages = 1

        regio: str = province[1]
        house_aptmnt: str = province[0]

        url: str = f"https://www.immoweb.be/en/search/{house_aptmnt}/for-sale/{regio}?countries=BE&page=1&orderBy=cheapest"

        pagination: int = [0]
        firefox_profile = webdriver.FirefoxProfile()
        firefox_profile.set_preference("permissions.default.image", 2)
        driver = webdriver.Firefox(firefox_profile)

        while True:
            pager = f"&page={num_pages}"
            url_paged = url + pager
            print(url_paged)
            if url_paged not in self.url_results_search:
                driver.get(url_paged)
                text = driver.page_source

                if "pagination__link pagination__link--next button button--text button--size-small" not in text:
                    print(url_paged, " pas end ou pas trouvé")
                    break

                a_elements = driver.find_elements_by_css_selector('a.card__title-link')
                for element in a_elements:
                    self.url_immo.add(element.get_attribute('href'))
                print(len(self.url_immo), url_paged, " adresses url de biens", pagination, num_pages)
                self.url_results_search.add(url_paged)

            num_pages += 1

        self._save_urls_results_to_csv()
        self._save_url_immo()
        driver.close()

    #  https://www.immoweb.be/fr/recherche/maison-et-appartement/a-vendre/liege/province?countries=BE&orderBy=relevance&page=1
    def _generator_db_url(self):
        with ThreadPoolExecutorWithQueueSizeLimit(maxsize=10, max_workers=10) as executor:
            executor.map(self._scan_page_list, self.provinces_house_apartement)

        print('pages de biens :', len(self.url_immo))
        print('pages de résultats :', len(self.url_results_search))

        finish = time.perf_counter()
        minutes = int(round(finish - self.start, 2) / 60)
        secondes = round(finish - self.start, 2) % 60
        print(f"fini selenium en {minutes} minutes et {secondes} secondes")

        duration = 0.3  # seconds
        freq = 440  # Hz
        os.system('play -nq -t alsa synth {} sine {}'.format(duration, freq))

        ask = input("Rerun ? (y/n)")
        rerun = ask != "n"
        print(rerun)
        if rerun:
            self._generator_db_url()
        else:
            self.cluster()

    def cluster(self):
        clusters = {x for x in self.url_immo if "new-real-estate-project" in x}
        len_before = [len(self.url_immo), len(clusters)]
        print(len(self.url_immo), len(clusters))
        future: list = []
        try:
            with ThreadPoolExecutorWithQueueSizeLimit(maxsize=20, max_workers=20) as executor:
                future = executor.map(self.find_url_in_clusters, clusters, timeout=40)
                print("ping")
                for list_of_url in future:
                    if list_of_url is not None:
                        for url in list_of_url:
                            self.url_immo.add(url)
                        # print(type(list_of_url), list_of_url)
        except concurrent.futures.TimeoutError:
            print(len_before)
            print('Time out error on cluster.', len(self.url_immo))

        print(len(self.url_immo))
        self._save_urls_results_to_csv()

        os.system('play -nq -t alsa synth {} sine {}'.format(0.3, 440))

        ask = input("Rerun ? (y/n)")
        rerun = ask != "n"
        print(rerun)
        if rerun:
            self.cluster()


if __name__ == "__main__":
    immoweb = Immoweb()
    immoweb.run()

# 11h36
# 1165 secondes selenium 1456
# 1852
# 2400
