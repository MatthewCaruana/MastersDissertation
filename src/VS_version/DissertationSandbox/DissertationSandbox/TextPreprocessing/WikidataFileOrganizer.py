import os
import bz2
import glob
import xml.etree.ElementTree as ET
import json


def main():
    location = "..\..\..\..\Data\WikiData"
    
    os.chdir(location)
    
    file_count = 0
    
    if not os.path.exists("chunks"):
        os.mkdir("chunks")
    
    for file in glob.glob("*.bz2"):
        file_pages = 0
    
        chunkname = lambda filecount: os.path.join("chunks","chunk-"+str(file_count)+".xml.bz2")
        chunkfile = bz2.BZ2File(chunkname(file_count), 'w')
    
        bz2file = bz2.BZ2File(file)
    
        current_page = ""
        page_content = []
        inMediaLink = False
        for line in bz2file:
            if '<page>' in line.decode('utf-8'):
                inMediaLink = True
    
            if inMediaLink:
                current_page = current_page + line.decode('utf-8')
                if '</page>' in line.decode('utf-8'):
                    root = ET.fromstring(current_page)
                    page_content.append(root.find("revision").find("text").text)
                    chunkfile.write(current_page.encode('utf-8'))
                    current_page = ""
                    file_pages += 1
                if file_pages > 1999:
                    chunkfile.close()
    
                    json_file = open("chunks/chunk-"+str(file_count)+".json", 'w')
                    json_file.write(json.dumps(page_content))
                    json_file.close()
                    page_content = []
    
                    file_pages = 0
                    file_count += 1
                    chunkfile = bz2.BZ2File(chunkname(file_count), 'w')
    
        chunkfile.close()
        file_count += 1

if __name__ == "__main__":
    main()