class CypherStringBuilder:
    def __init__(self):
        self.QueryString = ""

    def MatchRelation(self):
        self.QueryString += "MATCH (n)-[r]->(m)"

    def Where(self, query):
        self.QueryString += "WHERE " + query

    def And(self, query):
        self.QueryString += "AND " + query

    def Or(self, query):
        self.QueryString += "OR " + query

    def ReturnRelation(self):
        self.QueryString += "RETURN n,r,m"

    def Limit(self, limit):
        self.QueryString += "LIMIT " + str(limit)

    def Raw(self, statement):
        self.QueryString += statement

    def New(self):
        self.QueryString = ""