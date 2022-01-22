import json


class Triple():
    def __init__(self, Subject, Relation, Object):
        self.subject = Subject
        self.relation = Relation
        self.object = Object
        # TO ADD ORIGINAL SENTENCE

    def convert_to_json(self):
        dictionary = self.to_dict()

        jsonString = ""
        jsonString += json.dumps(dictionary)

        return jsonString

    def to_dict(self):
        return {
            'Subject': self.subject,
            'Relation': self.relation,
            'Object': self.object
        }