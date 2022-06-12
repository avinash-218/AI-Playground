'''
Disclaimer:

DeepSphere.AI developed these materials based on its team’s expertise and technical infrastructure, and we are sharing these materials strictly for learning and research.
These learning resources may not work on other learning infrastructures and DeepSphere.AI advises the learners to use these materials at their own risk. As needed, we will
be changing these materials without any notification and we have full ownership and accountability to make any change to these materials.

Author :                          Chief Architect :       Reviewer :
____________________________________________________________________________
Avinash R & Jothi Periasamy       Jothi Periasamy         Jothi Periasamy
'''

from rdflib import Graph

g = Graph()
g.parse("./Utility/DSAI_Kbot_Graph.ttl", format="turtle")

# returns total number of triples in the knowledge base
res = g.query("""
SELECT (COUNT(*) as ?triples)
	WHERE {
	    ?s ?p ?o
	}
""")
for row in res:
    print("Total number of triples in the knowledge base: " + row[0])

# returns total number of students
res = g.query("""
PREFIX ex: <http://example.org/>
SELECT (COUNT(?student) as ?count)
    WHERE {
        ?student ex:Occupation ex:STUDENT
    }
""")
for row in res:
    print("Total number of Students: " + row[0])

# returns total number of teachers
res = g.query("""
PREFIX ex: <http://example.org/>
SELECT (COUNT(?teacher) as ?count)
    WHERE {
        ?teacher ex:Occupation ex:TEACHER
    }
""")
for row in res:
    print("Total number of Teachers: " + row[0])

# returns total number of person working in deepsphere
res = g.query("""
PREFIX ex: <http://example.org/>
SELECT (COUNT(?deepsphere) as ?count)
    WHERE {
        ?deepsphere ex:WorksAt ex:DEEPSPHERE
    }
""")
for row in res:
    print("Total number of person working at DeepSphere: " + row[0])

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
		print("Yes, Both are Interns")
	else:
		print("No, Both are not Interns")

# returns students
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?student
    WHERE {
        ?student ex:Occupation ex:STUDENT
    }
""")
print("The Students are : ")
for row in res:
    print(row[0].toPython().split('/')[-1])
################################################
# returns mike's interest
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?interest
    WHERE {
        ex:MIKE ex:Interest ?interest
    }
""")
print("Mike's interest(s) is/are : ")
for row in res:
    print(row[0].toPython().split('/')[-1])

# returns everyone's interest
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?interest
    WHERE {
        ?s ex:Interest ?interest
    }
""")
print("Everyone has one or more interest in : ")
for row in res:
    print(row[0].toPython().split('/')[-1])

################################################
# returns mike's age
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?age
    WHERE {
        ex:MIKE ex:Age ?age
    }
""")
print("Mike's age is : ")
for row in res:
    print(row[0].toPython())

# returns everyone's age details
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name ?age
    WHERE {
        ?name ex:Age ?age
    }
""")
print("Age Details : ")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], '\tAge :', row[1].toPython())

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
    print('Name :',row[0].toPython().split('/')[-1], '\tCharacter :', row[1].toPython().split('/')[-1])

# returns everyone's completed degree
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name ?degree
    WHERE {
        ?name ex:Completed ?degree
    }
""")
print("Degree Details : ")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], '\tDegree :', row[1].toPython().split('/')[-1])

# returns everyone's interest
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name ?interest
    WHERE {
        ?name ex:Interest ?interest
    }
""")
print("Interests : ")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], '\tInterest :', row[1].toPython().split('/')[-1])

# returns everyone's project details
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name ?project
    WHERE {
        ?name ex:DoneProject ?project
    }
""")
print("Project Details : ")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], '\tProject :', row[1].toPython().split('/')[-1])

# returns everyone's working position
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name ?position
    WHERE {
        ?name ex:Position ?position
    }
""")
print("Working Position : ")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], '\tPosition :', row[1].toPython().split('/')[-1])

#who teaches John
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name
    WHERE {
        ?name ex:Teaches ex:JOHN
    }
""")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], 'teaches John')

#who teaches DS
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name
    WHERE {
        ?name ex:Teaches ex:DS
    }
""")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], 'teaches DS')

# who guided John
res = g.query("""
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?name
    WHERE {
        ?name ex:Guided ex:JOHN
    }
""")
for row in res:
    print('Name :',row[0].toPython().split('/')[-1], 'Guided John')

#friendship
res = g.query("""
PREFIX relation: <http://purl.org/vocab/relationship/>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX ex: <http://example.org/>
SELECT ?friend1 ?friend2
    WHERE {
        ?friend1 relation:friendOf ?friend2
    }
""")
for row in res:
    print(row[0].toPython().split('/')[-1], 'is a friend of', row[1].toPython().split('/')[-1])

'''
Copyright Notice:

Local and international copyright laws protect this material. Repurposing or reproducing this material without written approval from DeepSphere.AI violates the law.

(c) DeepSphere.AI
'''