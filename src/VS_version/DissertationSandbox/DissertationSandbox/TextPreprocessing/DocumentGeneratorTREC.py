import re
import json
import zeep

location = "data\\WashingtonPost\\data\\"

#Load file
file = open(location + "TREC_Washington_Post_collection.v3.jl", "r", encoding="utf-8")

wsdl = "http://metanet4u.research.um.edu.mt/services/MtPOS?wsdl"
client = zeep.Client(wsdl=wsdl)

all_content = []

html_compiler = re.compile("<.*?>")

count = 0
content_count = 0
for content in file:
    json_content = json.loads(content)
    line_contents = json_content["contents"]

    for section in line_contents:
        if section == None:
            continue
        if section["type"] == "sanitized_html":
            if "mime" in section:
                if section["mime"] == "text/html":
                    # remove html tags
                    section_content = re.sub(html_compiler, '', section["content"])
                else:
                    section_content = section["content"]

                #print(section_content)
                all_content.append(section_content)

                count = count + 1

        if count % 100000 == 0 and count > 0:
            writing_file = open(location + "TREC_Washington_Post_compiled_" + str(count) + ".txt", "w", encoding="utf-8")
            if len(all_content) > 0:
                for content in all_content:
                    writing_file.write(content + "\n")
            writing_file.close()

            all_content = []
    print(str(content_count) + " Processed")
    content_count = content_count + 1

file.close()

