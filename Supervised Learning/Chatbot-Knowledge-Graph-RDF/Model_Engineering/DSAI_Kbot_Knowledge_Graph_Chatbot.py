'''
Disclaimer:

DeepSphere.AI developed these materials based on its team’s expertise and technical infrastructure, and we are sharing these materials strictly for learning and research.
These learning resources may not work on other learning infrastructures and DeepSphere.AI advises the learners to use these materials at their own risk. As needed, we will
be changing these materials without any notification and we have full ownership and accountability to make any change to these materials.

Author :                          Chief Architect :       Reviewer :
____________________________________________________________________________
Avinash R & Jothi Periasamy       Jothi Periasamy         Jothi Periasamy
'''

from rdflib import Graph, Literal
import re

# generate graph of knowledge base
g = Graph()
g.parse("./Utility/DSAI_Kbot_Graph.ttl", format="turtle")

class Bot:
  def respond(self,str):
    if(re.search("(who are|name) the students[?]?", str)):
      # returns students
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?student
          WHERE {
              ?student ex:Occupation ex:STUDENT
          }
      """)
      print("The Students are : ", end='')
      for row in res:
          print(row[0].toPython().split('/')[-1], end='\t')
      print()

    elif(re.search('(how many students are there|what is the count of students)[?]?', str)):
      # returns total number of students
      res = g.query("""
      PREFIX ex: <http://example.org/>
      SELECT (COUNT(?student) as ?count)
          WHERE {
              ?student ex:Occupation ex:STUDENT
          }
      """)
      for row in res:
          print("There are", row[0], "Students")

    elif(re.search('(how many teachers are there|what is the count of teachers)[?]?', str)):
      # returns total number of teachers
      res = g.query("""
      PREFIX ex: <http://example.org/>
      SELECT (COUNT(?teacher) as ?count)
          WHERE {
              ?teacher ex:Occupation ex:TEACHER
          }
      """)
      for row in res:
          print("There are", row[0], "Teacher(s)")

    elif(re.search('how many (persons )?are working (at|in) deepsphere[?]?', str)):
      # returns total number of person working in deepsphere
      res = g.query("""
      PREFIX ex: <http://example.org/>
      SELECT (COUNT(?deepsphere) as ?count)
          WHERE {
              ?deepsphere ex:WorksAt ex:DEEPSPHERE
          }
      """)
      for row in res:
          print("There are", row[0], "person(s) working at DeepSphere")

    elif(re.search('are the (both )?students interns at deepsphere[?]?', str)):
      # return if both are interns
      res = g.query("""
      PREFIX ex: <http://example.org/>
      SELECT (COUNT(?intern) as ?count)
          WHERE {
              ?intern ex:Position ex:INTERN
          }
      """)
      for row in res:
        if(row[0].toPython()==2):
          print("Yes, Both the students are Interns")
        else:
          print("No, Both the students are not Interns")

    elif(re.search('what interest (each has|does they have)[?]?', str)):
      # returns everyone's interest
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?interest
          WHERE {
              ?name ex:Interest ?interest
          }
      """)
      for row in res:
          print('Name :',row[0].toPython().split('/')[-1], '\tInterest :', row[1].toPython().split('/')[-1])

    elif(re.search('how old (is everyone|are they)[?]?', str)):
      # returns everyone's age details
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?age
          WHERE {
              ?name ex:Age ?age
          }
      """)
      for row in res:
          print(row[0].toPython().split('/')[-1], 'is', row[1].toPython(), 'years old')

    elif(re.search('what (are their characteristics|characteristic does they possess)[?]?', str)):
      # returns everyone's character
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?character
          WHERE {
              ?name ex:Character ?character
          }
      """)
      print("Character Details : ")
      for row in res:
          print(row[0].toPython().split('/')[-1], 'is', row[1].toPython().split('/')[-1])

    elif(re.search('what are their (highest )?qualifications[?]?', str)):
      # returns everyone's completed degree
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?degree
          WHERE {
              ?name ex:Completed ?degree
          }
      """)
      for row in res:
          print(row[0].toPython().split('/')[-1], "'s highest qualification is ", row[1].toPython().split('/')[-1], sep='')

    elif(re.search('what (projects have the students done|are the projects of the students)[?]?', str)):
      # returns everyone's project details
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?project
          WHERE {
              ?name ex:DoneProject ?project
          }
      """)
      for row in res:
          print(row[0].toPython().split('/')[-1], 'did a project titled -', row[1].toPython().split('/')[-1])

    elif(re.search('what are their (position at workplace|working position)[?]?', str)):
      # returns everyone's working position
      res = g.query("""
      PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
      PREFIX ex: <http://example.org/>
      SELECT ?name ?position
          WHERE {
              ?name ex:Position ?position
          }
      """)
      for row in res:
          print(row[0].toPython().split('/')[-1], 'is a', row[1].toPython().split('/')[-1])

    elif(str=='quit'):
      print('Bye...')
    else:
      print("Sorry, I can't get you... Try again!!!")

def main():
  print('-' * 100)
  print("Hi!!! I'm Jothi, DeepSphere's Chatbot. Enter 'quit' to exit")
  print('-'*100)

  s = ''
  chatbot = Bot();
  while s != 'quit':
    try:
      s = input('> ').lower()
    except EOFError:
      s = 'quit'
    while s[-1] in '!.':
      s = s[:-1]
    chatbot.respond(s)


if __name__ == "__main__":
  main()

'''
Copyright Notice:

Local and international copyright laws protect this material. Repurposing or reproducing this material without written approval from DeepSphere.AI violates the law.

(c) DeepSphere.AI
'''