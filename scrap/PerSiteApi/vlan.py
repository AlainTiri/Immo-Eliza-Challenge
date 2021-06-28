import time
from random import random

import requests
from bs4 import BeautifulSoup
import os
import re
import pandas as pd
import scrapy
import csv
from lxml import etree

# Url of website
from pandas.io.common import urlopen

url = 'https://www.2ememain.be/l/immo/'
other_url = 'https://immo.vlan.be/fr/immobilier?propertytypes=appartement,maison,business,terrain,garage,' \
            'bien-d%27investissement,kot,divers&transactiontypes=a-vendre,a-louer,en-vente-publique,' \
            'en-colocation&propertysubtypes=appartement,rez-de-chaussee,duplex,penthouse,studio,loft,triplex,maison,' \
            'villa,immeuble-mixte,maison-de-maitre,fermette,bungalow,chalet,chateau,surface-commerciale,' \
            'bureaux,surface-industrielle,fonds-de-commerce,ferme,terrain-a-batir,terrain,terrain-agricole,' \
            'parking,garage,maison-de-rapport,kot-etudiant,bien-divers&countries=belgique&noindex=1'
user_agent_desktop = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) ' \
                     'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.149 ' \
                     'blob/537.36'

url_page_offset = '&pageOffset='

headers = {'User-Agent': user_agent_desktop}

__per_site_api_path = os.path.dirname(__file__)
__root_path = os.path.dirname(__per_site_api_path)
__core_path = os.path.join(__root_path, "Core")
__data_path = os.path.join(__root_path, "Data")
__database_file = os.path.join(__data_path, "database.csv")
__zipcode_file = os.path.join(__data_path, "zipcodes.csv")
__vlan_url = os.path.join(__data_path, "vlan.csv")
__vlan_link = os.path.join(__data_path, "vlan_link.csv")
__vlan_data = os.path.join(__data_path, "vlan_data.csv")
__root_url = r"https://immo.vlan.be"
__addresses = set()
__pattern = re.compile(r'/en/(buy|rent)/.+/[a-zA-Z]+-[0-9]+/.+.html')

__zipcode_df = pd.read_csv(__zipcode_file)
r = requests.get(other_url, headers=headers)
soup = BeautifulSoup(r.content, 'lxml')
for elem in soup.find_all('article', attrs={"class": "list-view-item"}):
    print(elem.get('href'))

""" propertytype"""
page_list = []
list_column = []
htmlparser = etree.HTMLParser()

nb_of_rows = 0
"""

/fr/detail/appartement/a-vendre/9600/renaix/raw59914
/fr/detail/immeuble-mixte/a-vendre/7120/peissant/var57020
"""
with open(__vlan_url, 'a+') as url_file:
    with open(__vlan_link, 'w+') as url_index_file:
        with open(__vlan_data, 'w+') as url_data_file:
            for offset in range(1, 11): #168
                link_list = []
                offset_url = other_url + url_page_offset + str(offset)
                r = requests.get(offset_url,
                                 headers=headers)  # http://validate.perfdrive.com/immovlan/captcha?ssa=caee399c-1aad-40c8-b615-a7fc6af6c779&ssb=mh0pep5i6pihbb24dm3h0k11k&ssc=http%3a%2f%2fimmo.vlan.be%2ffr%2fimmobilier%3fpropertytypes%3dappartement%2cmaison%2cbusiness%2cterrain%2cgarage%2cbien-d%2527investissement%2ckot%2cdivers%26transactiontypes%3da-vendre%2ca-louer%2cen-vente-publique%2cen-colocation%26propertysubtypes%3dappartement%2crez-de-chaussee%2cduplex%2cpenthouse%2cstudio%2cloft%2ctriplex%2cmaison%2cvilla%2cimmeuble-mixte%2cmaison-de-maitre%2cfermette%2cbungalow%2cchalet%2cchateau%2csurface-commerciale%2cbureaux%2csurface-industrielle%2cfonds-de-commerce%2cferme%2cterrain-a-batir%2cterrain%2cterrain-agricole%2cparking%2cgarage%2cmaison-de-rapport%2ckot-etudiant%2cbien-divers%26countries%3dbelgique%26noindex%3d1%26pageOffset%3d1&ssd=970959273986341&sse=olapjpemfpm@ggc&ssf=eb0f8f4c4fcb66138e2a0b11b2df4950e89d1b7e&ssg=2faf2114-77e1-4b80-b1e9-4d0c95be0d68&ssh=f205d560-d340-40d2-9250-da0d2d778adb&ssi=971a1329-be01-ef47-d925-95a95e54fd43&ssj=4e22da2e-2c6e-424e-a5dc-de03652e22d6&ssk=Immovlan-support@shieldsquare.com&ssl=970959273986341&ssm=97095927398634125109709592739863&ssn=eb0f8f4ca5d1f5d57d8406ea13b0eb0f8f4ceb0f8&sso=eb0f8eb0f8f4c00a330b2f5dce4a1fe5eeb0f8f4c&ssp=970959273997095162029709592739&ssq=970959210942970959273970959273986341&ssr=OTQuMTA0LjE5Ny4yMTM=&sss=Mozilla/5.0%20(compatible;%20Yahoo!%20Slurp;%20http://help.yahoo.com/help/us/ysearch/slurp)&sst=Mozilla/5.0%20(Windows%20NT%2010.0;%20Win64;%20x64)%20AppleWebKit/537.36%20(KHTML,%20like%20Gecko)%20Chrome/80.0.3987.149%20Safiri/537.36&ssu=Mozilla/5.0%20(compatible;%20Yahoo!%20Slurp;%20http://help.yahoo.com/help/us/ysearch/slurp)&ssv=o3pp9pv@2pl342s&ssw=&ssx=970959273986341&ssy=olapjpemfpm@ggcdiod@albblcnpekfaojknblho&ssz=eb0f8f4c4fcb661
                if "http://validate.perfdrive.com/immovlan/captcha" in r.url:
                    print("captchah")
                    print(r.url)
                    exit(0)
                url_index_file.write(str(r.status_code))
                url_index_file.write(offset_url)
                soup = BeautifulSoup(r.content, 'lxml')
                for soup_links in soup.findAll('a'):
                    link = soup_links.get("href")
                    if 'a-vendre' in link and '/fr/detail/' in link:
                        link_list.append(link)
                for links in sorted(link_list):
                    url_file.write(links)
                    url_file.write("\n")
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
#                            url_data_file.write(column_name.text.strip())
#                            url_data_file.write("\n")

                url_index_file.write("\n")
            for column in list_column:
                url_data_file.write(column)