import sys
import io
import re
import wikipedia
from tqdm import tqdm
import json

year_re = re.compile(r"\b[12][0-9]{3}\b")

wikipedia.set_lang("en")
wikipedia.set_rate_limiting(False)

in_fname = sys.argv[1]
out_fname = sys.argv[2]

in_f = io.open(in_fname)
out_f = open(out_fname, "wb")

#list of all PERSON NE
#nes = ["Dorfer", "Mr. J. C. Rastrick","Masaaki Shirakawa"]

#example of summary with var1 and var2 (for loop needed)
found_nes = {}


def page_years(page_title, depth=0):
    if depth > 5:
        return []
    try:
        page = wikipedia.page(page_title).content
        return year_re.findall(page)
    except wikipedia.DisambiguationError as e:
        years = []
        disamb = str(e).split("\n")[1:6]
        for title in disamb:
            years.extend(page_years(title, depth + 1))
        return years
    except:
        return []


for line in tqdm(in_f.readlines()):
    ne_str = line.strip().split('\t')[1]
    years = []
    matches = wikipedia.search(ne_str, results=5)
    for match in matches:
        years.extend(page_years(match))

    found_nes[ne_str] = years

json.dump(found_nes, out_f)

in_f.close()
out_f.close()
