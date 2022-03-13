import requests
import json

class MAPAWrapper():
    def __init__(self):
        self.base_url = "http://localhost:8671/mapa/1.0/anonymization/anonymize_text"
        self.headers = {'Content-Type': 'application/json'}

    def process(self, sentence, operation):
        sentence_content = {"text": sentence, "operation": operation }

        r = requests.post(self.base_url, headers=self.headers, json=sentence_content)
        return self.reformat(r)
    
    def reformat(self, response):
        annotated_content = json.loads(response.content)["annotations"]
        return annotated_content

    @staticmethod
    def get_dates(annotations):
        dates = []
        date_annotations = ["DATE", "TIME", "day", "day of week", "month", "other:date", "unresolved:date", "year"]

        for annotation in annotations:
            if annotation["value"] in date_annotations:
                dates.append([annotation["content"], annotation["value"]])

        return dates


if __name__ == "__main__":
    mapa_wrapper = MAPAWrapper()
    mapa_wrapper.process("El señor Pérez.", "NOOP")