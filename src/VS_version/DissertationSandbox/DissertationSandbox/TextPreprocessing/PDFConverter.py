
import fitz
import os

def fileTransformer(file_locations):
    for file_location in file_locations:
        with fitz.open(file_location) as doc:
            text= ""
            for page in doc:
                text += page.get_text()
                
            file = open(file_location.replace("pdf", "txt"), "a", encoding='utf-8')
            file.write(text)
            file.close()

def main():
    files_english = os.listdir('..\..\..\..\Data\MaltaBudgets\English')
    files_maltese = os.listdir('..\..\..\..\Data\MaltaBudgets\Maltese')
    
    file_locations_english = []
    for file in files_english:
        file_locations_english.append(os.path.abspath('..\..\..\..\Data\MaltaBudgets\English\\' + file))
    
    file_locations_maltese = []
    for file in files_maltese:
        file_locations_maltese.append(os.path.abspath('..\..\..\..\Data\MaltaBudgets\Maltese\\' + file))
    
    
    fileTransformer(file_locations_english)
    #fileTransformer(file_locations_maltese)

if __name__ == "__main__":
    main()