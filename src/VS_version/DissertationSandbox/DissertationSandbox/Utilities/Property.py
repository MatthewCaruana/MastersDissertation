import json


class Property():
    def __init__(self, name, text):
        self.Name = name
        self.Text = text

    def convert_to_json(self):
        dictionary = self.to_dict()

        jsonString = ""
        jsonString += json.dumps(dictionary)

        return jsonString

    def to_dict(self):
        return {
            self.Name: self.Text
        }